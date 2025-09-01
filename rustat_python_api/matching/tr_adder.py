import pandas as pd
import numpy as np


class TrackingFeaturesAdder:
    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame, ball: pd.DataFrame):
        self.events = events
        self.tracking = tracking
        self.ball = ball

        # Normalize dtypes
        if 'team_id' in self.tracking.columns:
            self.tracking['team_id'] = self.tracking['team_id'].astype(str)
        if 'team_id' in self.events.columns:
            # keep original, but store string view for joins/masks
            self._events_team_str = self.events['team_id'].astype(str)
        else:
            self._events_team_str = pd.Series(index=self.events.index, dtype=str)

        # Cache team ids for side mapping
        self.team_ids = list(self.tracking['team_id'].dropna().astype(str).unique())

        # Build side_by_half using tracking side_1h
        # sides: Series index team_id -> np.ndarray like ["left"] or ["right"]
        sides = self.tracking.groupby('team_id')['side_1h'].unique()
        # Align to our team_ids and take the first value
        sides = sides.reindex(self.team_ids)
        side_by_team = {
            tid: (arr[0] if isinstance(arr, (list, np.ndarray)) and len(arr) > 0 else np.nan)
            for tid, arr in sides.items()
        }
        # Half 2 sides are swapped
        self.side_by_half = {
            1: side_by_team,
            2: {
                tid: ('left' if side == 'right' else 'right') if isinstance(side, str) else np.nan
                for tid, side in side_by_team.items()
            }
        }

    @staticmethod
    def to_tracking_second(sec: float) -> float:
        return float(np.round(np.rint(sec * 30.0) / 30.0, 2))

    def _build_frames(self) -> dict:
        """
        Build dictionary: (half, sec_tracking) -> dict with arrays: x, y, team_id(str), player_id
        """
        tr = self.tracking.copy()
        # Ensure numeric positions
        tr['pos_x'] = pd.to_numeric(tr['pos_x'], errors='coerce')
        tr['pos_y'] = pd.to_numeric(tr['pos_y'], errors='coerce')
        tr['sec_tracking'] = tr['second'].apply(self.to_tracking_second)

        frames = {}
        cols_needed = ['team_id', 'pos_x', 'pos_y']
        if 'player_id' in tr.columns:
            cols_needed.append('player_id')
        else:
            tr['player_id'] = -1
            cols_needed.append('player_id')

        for (half, sec), df in tr.groupby(['half', 'sec_tracking'], sort=False):
            frames[(int(half), float(sec))] = {
                'team': df['team_id'].astype(str).to_numpy(),
                'x': df['pos_x'].to_numpy(dtype=float),
                'y': df['pos_y'].to_numpy(dtype=float),
                'pid': df['player_id'].to_numpy(),
            }
        return frames

    @staticmethod
    def _flip_coords(x: np.ndarray, y: np.ndarray, do_flip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Flip in TV orientation box 105x68
        xf = np.where(do_flip, 105.0 - x, x)
        yf = np.where(do_flip,  68.0 - y, y)
        return xf, yf

    def run(self, radii: list[float], cone_degrees: list[float] | None = None, k_list: list[int] | None = None) -> None:
        """
        Compute tracking-based features for each event and write them into self.events.

        Features (for pos and pos_dest, unless noted otherwise):
        - counts of teammates/opponents with x <, x > current point (attack-aligned)
        - counts of teammates/opponents within given radii
        - pos only: x of second-back (closest to own goal excluding GK via 2nd) and farthest x
        """
        # Prepare events times aligned to tracking
        self.events['sec_tracking'] = self.events['second'].map(self.to_tracking_second)

        # Build right-side sets per half for flipping event coordinates
        right_by_half = {
            int(h): {tid for tid, side in sides.items() if side == 'right'}
            for h, sides in self.side_by_half.items()
        }

        # Event coordinate arrays
        n = self.events.index.size
        halves_arr = self.events['half'].to_numpy()
        teams_ev_str = self._events_team_str.to_numpy()
        x = self.events['pos_x'].to_numpy()
        y = self.events['pos_y'].to_numpy()
        xd = self.events['pos_dest_x'].to_numpy()
        yd = self.events['pos_dest_y'].to_numpy()

        # Flip mask for events where the event team plays RIGHT in this half (align to TV orientation)
        flip_mask = np.zeros(n, dtype=bool)
        for h, right_set in right_by_half.items():
            if not right_set:
                continue
            idx_h = (halves_arr == h)
            idx_right = np.isin(teams_ev_str, list(right_set))
            flip_mask |= (idx_h & idx_right)

        x, y = self._flip_coords(x, y, flip_mask)
        xd, yd = self._flip_coords(xd, yd, flip_mask)

        # Masks
        src_nan = np.isnan(x) | np.isnan(y)
        dest_nan = np.isnan(xd) | np.isnan(yd)
        eq_eps = 1e-6
        eq_mask = (~dest_nan) & np.isclose(xd, x, atol=eq_eps) & np.isclose(yd, y, atol=eq_eps)

        # Build tracking frames map
        frames = self._build_frames()
        secs = self.events['sec_tracking'].to_numpy()
        pids = self.events['player_id'].to_numpy() if 'player_id' in self.events.columns else np.full(n, -1)

        # Preallocate outputs
        # Source basic counts
        src_tm_lt_x = np.full(n, np.nan)
        src_tm_gt_x = np.full(n, np.nan)
        src_opp_lt_x = np.full(n, np.nan)
        src_opp_gt_x = np.full(n, np.nan)

        # Dest basic counts
        dst_tm_lt_x = np.full(n, np.nan)
        dst_tm_gt_x = np.full(n, np.nan)
        dst_opp_lt_x = np.full(n, np.nan)
        dst_opp_gt_x = np.full(n, np.nan)

        # Radii counts: dict rad -> arrays
        radii = list(radii or [])
        src_tm_in_r = {rad: np.full(n, np.nan) for rad in radii}
        src_opp_in_r = {rad: np.full(n, np.nan) for rad in radii}
        dst_tm_in_r = {rad: np.full(n, np.nan) for rad in radii}
        dst_opp_in_r = {rad: np.full(n, np.nan) for rad in radii}

        # Cone counts by degrees
        cone_degrees = list(cone_degrees or [])
        src_tm_cone = {deg: np.full(n, np.nan) for deg in cone_degrees}
        src_opp_cone = {deg: np.full(n, np.nan) for deg in cone_degrees}
        dst_tm_cone = {deg: np.full(n, np.nan) for deg in cone_degrees}
        dst_opp_cone = {deg: np.full(n, np.nan) for deg in cone_degrees}

        # k-NN mean distances
        k_list = list(k_list or [])
        src_tm_mean_k = {k: np.full(n, np.nan) for k in k_list}
        src_opp_mean_k = {k: np.full(n, np.nan) for k in k_list}
        dst_tm_mean_k = {k: np.full(n, np.nan) for k in k_list}
        dst_opp_mean_k = {k: np.full(n, np.nan) for k in k_list}

        # Pos-only opponent extremes
        src_opp_second_back_x = np.full(n, np.nan)
        src_opp_farthest_x = np.full(n, np.nan)

        for i in range(n):
            half = int(halves_arr[i])
            sec = float(secs[i])
            team_i = teams_ev_str[i]
            pid_i = pids[i]

            frame = frames.get((half, sec))
            if frame is None:
                continue

            teams_f = frame['team']
            xs = frame['x']
            ys = frame['y']
            pids_f = frame['pid']

            # Masks relative to event team
            tm_mask = (teams_f == team_i)
            opp_mask = ~tm_mask

            # Source position features
            if not src_nan[i]:
                xi, yi = float(x[i]), float(y[i])

                # Align x with attack direction: for right-side team, ahead is smaller TV-x
                side_x = self.side_by_half.get(half, {}).get(team_i, np.nan)
                sign = 1.0 if side_x == 'left' else (-1.0 if side_x == 'right' else 1.0)
                src_tm_lt_x[i] = np.sum(tm_mask & ((sign * (xs - xi)) < 0))
                src_tm_gt_x[i] = np.sum(tm_mask & ((sign * (xs - xi)) > 0))

                src_opp_lt_x[i] = np.sum(opp_mask & ((sign * (xs - xi)) < 0))
                src_opp_gt_x[i] = np.sum(opp_mask & ((sign * (xs - xi)) > 0))

                if radii:
                    d2 = (xs - xi) ** 2 + (ys - yi) ** 2
                    not_self = (pids_f != pid_i)
                    for rad in radii:
                        mask_r = d2 <= (float(rad) * float(rad))
                        src_tm_in_r[rad][i] = np.sum(tm_mask & not_self & mask_r)
                        src_opp_in_r[rad][i] = np.sum(opp_mask & mask_r)

                # Cones (pos): counts within +/- deg around forward-to-goal direction
                if cone_degrees:
                    side = self.side_by_half.get(half, {}).get(team_i, np.nan)
                    if isinstance(side, str):
                        f_sign = 1.0 if side == 'left' else -1.0  # +x if left-side team, else -x
                        dx = xs - xi
                        dy = ys - yi
                        d = np.sqrt(dx * dx + dy * dy) + 1e-8
                        cosang = (f_sign * dx) / d  # dot with unit forward vector
                        not_self = (pids_f != pid_i)
                        for deg in cone_degrees:
                            cth = np.cos(np.deg2rad(float(deg)))
                            in_cone = cosang >= cth
                            src_tm_cone[deg][i] = np.sum(in_cone & tm_mask & not_self)
                            src_opp_cone[deg][i] = np.sum(in_cone & opp_mask)

                # k-NN mean distances (pos)
                if k_list:
                    dx = xs - xi
                    dy = ys - yi
                    d = np.sqrt(dx * dx + dy * dy)
                    # teammates excluding self
                    mask_tm_valid = tm_mask & (pids_f != pid_i)
                    dist_tm = d[mask_tm_valid]
                    dist_opp = d[opp_mask]
                    dist_tm.sort()
                    dist_opp.sort()
                    for k in k_list:
                        if dist_tm.size > 0:
                            src_tm_mean_k[k][i] = float(np.mean(dist_tm[:min(k, dist_tm.size)]))
                        if dist_opp.size > 0:
                            src_opp_mean_k[k][i] = float(np.mean(dist_opp[:min(k, dist_opp.size)]))

                # Opponent extremes (TV orientation)
                side = self.side_by_half.get(half, {}).get(team_i, np.nan)
                if isinstance(side, str):
                    # Extremes in TV orientation (left-to-right)
                    x_opp_tv = xs[opp_mask]
                    if x_opp_tv.size >= 2:
                        opp_side = 'left' if side == 'right' else 'right'
                        if opp_side == 'left':
                            x_sorted = np.sort(x_opp_tv)
                            src_opp_farthest_x[i] = float(np.max(x_opp_tv))
                            if x_sorted.size >= 2:
                                src_opp_second_back_x[i] = float(x_sorted[1])  # 2nd smallest
                        else:  # opp_side == 'right'
                            x_sorted = np.sort(x_opp_tv)[::-1]
                            src_opp_farthest_x[i] = float(np.min(x_opp_tv))
                            if x_sorted.size >= 2:
                                src_opp_second_back_x[i] = float(x_sorted[1])  # 2nd largest

            # Destination position features
            if eq_mask[i]:
                # copy from source if dest coincides with source
                dst_tm_lt_x[i] = src_tm_lt_x[i]
                dst_tm_gt_x[i] = src_tm_gt_x[i]
                dst_opp_lt_x[i] = src_opp_lt_x[i]
                dst_opp_gt_x[i] = src_opp_gt_x[i]
                for rad in radii:
                    dst_tm_in_r[rad][i] = src_tm_in_r[rad][i]
                    dst_opp_in_r[rad][i] = src_opp_in_r[rad][i]
            elif not dest_nan[i]:
                xdi, ydi = float(xd[i]), float(yd[i])

                side_x = self.side_by_half.get(half, {}).get(team_i, np.nan)
                sign = 1.0 if side_x == 'left' else (-1.0 if side_x == 'right' else 1.0)
                dst_tm_lt_x[i] = np.sum(tm_mask & ((sign * (xs - xdi)) < 0))
                dst_tm_gt_x[i] = np.sum(tm_mask & ((sign * (xs - xdi)) > 0))

                dst_opp_lt_x[i] = np.sum(opp_mask & ((sign * (xs - xdi)) < 0))
                dst_opp_gt_x[i] = np.sum(opp_mask & ((sign * (xs - xdi)) > 0))

                if radii:
                    d2d = (xs - xdi) ** 2 + (ys - ydi) ** 2
                    not_self = (pids_f != pid_i)
                    for rad in radii:
                        mask_rd = d2d <= (float(rad) * float(rad))
                        dst_tm_in_r[rad][i] = np.sum(tm_mask & not_self & mask_rd)
                        dst_opp_in_r[rad][i] = np.sum(opp_mask & mask_rd)

                # Cones (dest)
                if cone_degrees:
                    side = self.side_by_half.get(half, {}).get(team_i, np.nan)
                    if isinstance(side, str):
                        f_sign = 1.0 if side == 'left' else -1.0
                        dx = xs - xdi
                        dy = ys - ydi
                        d = np.sqrt(dx * dx + dy * dy) + 1e-8
                        cosang = (f_sign * dx) / d
                        not_self = (pids_f != pid_i)
                        for deg in cone_degrees:
                            cth = np.cos(np.deg2rad(float(deg)))
                            in_cone = cosang >= cth
                            dst_tm_cone[deg][i] = np.sum(in_cone & tm_mask & not_self)
                            dst_opp_cone[deg][i] = np.sum(in_cone & opp_mask)

                # k-NN mean distances (dest)
                if k_list:
                    dx = xs - xdi
                    dy = ys - ydi
                    d = np.sqrt(dx * dx + dy * dy)
                    mask_tm_valid = tm_mask & (pids_f != pid_i)
                    dist_tm = d[mask_tm_valid]
                    dist_opp = d[opp_mask]
                    dist_tm.sort()
                    dist_opp.sort()
                    for k in k_list:
                        if dist_tm.size > 0:
                            dst_tm_mean_k[k][i] = float(np.mean(dist_tm[:min(k, dist_tm.size)]))
                        if dist_opp.size > 0:
                            dst_opp_mean_k[k][i] = float(np.mean(dist_opp[:min(k, dist_opp.size)]))

        # Write columns to events
        self.events['tf_src_tm_lt_x'] = src_tm_lt_x
        self.events['tf_src_tm_gt_x'] = src_tm_gt_x
        self.events['tf_src_opp_lt_x'] = src_opp_lt_x
        self.events['tf_src_opp_gt_x'] = src_opp_gt_x

        self.events['tf_dest_tm_lt_x'] = dst_tm_lt_x
        self.events['tf_dest_tm_gt_x'] = dst_tm_gt_x
        self.events['tf_dest_opp_lt_x'] = dst_opp_lt_x
        self.events['tf_dest_opp_gt_x'] = dst_opp_gt_x

        for rad in radii:
            r_tag = f"r{float(rad):g}"
            self.events[f'tf_src_tm_in_{r_tag}'] = src_tm_in_r[rad]
            self.events[f'tf_src_opp_in_{r_tag}'] = src_opp_in_r[rad]
            self.events[f'tf_dest_tm_in_{r_tag}'] = dst_tm_in_r[rad]
            self.events[f'tf_dest_opp_in_{r_tag}'] = dst_opp_in_r[rad]

        # Cone features
        for deg in cone_degrees:
            d_tag = f"deg{float(deg):g}"
            self.events[f'tf_src_tm_cone_{d_tag}'] = src_tm_cone[deg]
            self.events[f'tf_src_opp_cone_{d_tag}'] = src_opp_cone[deg]
            self.events[f'tf_dest_tm_cone_{d_tag}'] = dst_tm_cone[deg]
            self.events[f'tf_dest_opp_cone_{d_tag}'] = dst_opp_cone[deg]

        # k-NN mean distances
        for k in k_list:
            self.events[f'tf_src_tm_mean_dist_k{k}'] = src_tm_mean_k[k]
            self.events[f'tf_src_opp_mean_dist_k{k}'] = src_opp_mean_k[k]
            self.events[f'tf_dest_tm_mean_dist_k{k}'] = dst_tm_mean_k[k]
            self.events[f'tf_dest_opp_mean_dist_k{k}'] = dst_opp_mean_k[k]

        # Pos-only opponent extremes
        self.events['tf_src_opp_second_back_x'] = src_opp_second_back_x
        self.events['tf_src_opp_farthest_x'] = src_opp_farthest_x
