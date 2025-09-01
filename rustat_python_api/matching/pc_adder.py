from rustat_python_api import PitchControl
import pandas as pd
import numpy as np


class PitchControlAdder:
    def __init__(
            self,
            events: pd.DataFrame, tracking: pd.DataFrame, ball: pd.DataFrame,
            device: str = 'cpu', backend: str = 'pt'
    ):
        self.events = events
        self.tracking = tracking
        self.ball = ball

        self.device = device
        self.backend = backend

        self.pitch_control = PitchControl(tracking, events, ball)
        self.sec2timestamp = self._get_sec2timestamp()

    @staticmethod
    def to_tracking_second(sec: float) -> float:
        return float(np.round(np.rint(sec * 30.0) / 30.0, 2))

    def _get_sec2timestamp(self):
        return {
            half: {val: i for i, val in enumerate(self.pitch_control.t[half].tolist())}
            for half in self.pitch_control.t.keys()
        }

    def _get_pc(self, sec: float, half: int, dt: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamp = self.sec2timestamp[half][sec]
        pc, xx, yy = self.pitch_control.fit(half=half, tp=timestamp, dt=dt, backend=self.backend, device=str(self.device))

        return pc, xx, yy

    @staticmethod
    def _get_pc_value_nearest(
            pc, xx, yy,
            x: float, y: float
    ) -> float:
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        i, j = np.unravel_index(np.argmin(d2), pc.shape)

        return float(pc[i, j])

    def _get_pc_value_mean(
            self,
            pc, xx, yy,
            x: float, y: float, rad: float
    ) -> float:
        if rad is None or rad <= 0:
            return self._get_pc_value_nearest(pc, xx, yy, x, y)

        d2 = (xx - x) ** 2 + (yy - y) ** 2
        mask = d2 <= (rad * rad)

        if not np.any(mask):
            return self._get_pc_value_nearest(pc, xx, yy, x, y)

        return float(pc[mask].mean())

    def _get_pc_value_gaussian(
            self,
            pc, xx, yy,
            x: float, y: float, sigma: float
    ) -> float:
        if sigma is None or sigma <= 0:
            return self._get_pc_value_nearest(pc, xx, yy, x, y)

        d2 = (xx - x) ** 2 + (yy - y) ** 2
        w = np.exp(-d2 / (2.0 * sigma * sigma))
        z = w.sum()

        if z <= 0:
            return self._get_pc_value_nearest(pc, xx, yy, x, y)

        return float((pc * w).sum() / z)

    def _get_pc_value(
            self,
            pc: np.ndarray, xx: np.ndarray, yy: np.ndarray,
            x: float, y: float, rad: float, mode: str = 'mean'
    ) -> float:
        """
        mode: one of 'mean', 'sum', 'mass', 'gaussian', 'nearest'
        """
        match mode:
            case 'mean':
                return self._get_pc_value_mean(pc, xx, yy, x, y, rad)
            # case 'sum':
            #     return self._get_pc_value_sum(pc, xx, yy, x, y, rad)
            # case 'mass':
            #     return self._get_pc_value_mass(pc, xx, yy, x, y, rad)
            case 'gaussian':
                return self._get_pc_value_gaussian(pc, xx, yy, x, y, rad)
            case _:
                return self._get_pc_value_nearest(pc, xx, yy, x, y)

    def run(
            self,
            modes: list[str], rads: list[float], dt: int = 100
    ):
        assert len(modes) == len(rads)

        self.events['sec_tracking'] = self.events['second'].map(self.to_tracking_second)

        right_by_half = {
            int(h): {
                int(tid)
                for tid, side in self.pitch_control.side_by_half.get(int(h), {}).items()
                if side == 'right'
            }
            for h in self.pitch_control.side_by_half.keys()
        }

        # Preallocate outputs: for each (mode, rad) keep src and dest arrays
        n = self.events.index.size
        pcs_src = [np.full(n, np.nan, dtype=float) for _ in modes]
        pcs_dst = [np.full(n, np.nan, dtype=float) for _ in modes]

        # Vectorized flip mask: team plays to the right in this half
        flip_mask = np.zeros(n, dtype=bool)
        halves_arr = self.events['half'].to_numpy()
        team_series = self.events['team_id']

        for h, right_set in right_by_half.items():
            if not right_set:
                continue

            idx_h = (halves_arr == h)
            idx_right = team_series.isin(right_set).to_numpy()
            flip_mask |= (idx_h & idx_right)

        # Coordinates (can contain NaN)
        x = self.events['pos_x'].to_numpy()
        y = self.events['pos_y'].to_numpy()
        xd = self.events['pos_dest_x'].to_numpy()
        yd = self.events['pos_dest_y'].to_numpy()

        # Flip coordinates where necessary
        x = np.where(flip_mask, 105.0 - x, x)
        y = np.where(flip_mask,  68.0 - y, y)
        xd = np.where(flip_mask, 105.0 - xd, xd)
        yd = np.where(flip_mask,  68.0 - yd, yd)

        # Masks
        src_nan = np.isnan(x) | np.isnan(y)
        dest_nan = np.isnan(xd) | np.isnan(yd)
        eq_eps = 1e-6
        eq_mask = (~dest_nan) & np.isclose(xd, x, atol=eq_eps) & np.isclose(yd, y, atol=eq_eps)

        # Cache (half, sec) -> (pc, xx, yy)
        cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        secs = self.events['sec_tracking'].to_numpy()

        # for i in tqdm(range(n)):
        for i in range(n):
            sec = float(secs[i])
            half = int(halves_arr[i])

            # Ensure second exists in mapping for this half
            sec_map = self.sec2timestamp.get(half)
            if not sec_map or sec not in sec_map:
                print(f"[WARNING] Second {sec} not found for half {half}")
                continue

            key = (half, sec)
            if key in cache:
                pc, xx, yy = cache[key]
            else:
                try:
                    pc, xx, yy = self._get_pc(sec=sec, half=half, dt=dt)
                except Exception:
                    print(f"[WARNING] Failed to get PC for half {half}, second {sec}")
                    pc, xx, yy = (None, None, None)
                cache[key] = (pc, xx, yy)

            if pc is None:
                print(f"[WARNING] PC is None for half {half}, second {sec}")
                continue

            team = team_series.iloc[i]
            is_right_start = team in right_by_half.get(1, set())
            # if not is_right_start:
            #     pc = 1 - pc

            # Source
            if not src_nan[i]:
                xi = float(x[i])
                yi = float(y[i])
                for k, (mode, rad) in enumerate(zip(modes, rads)):
                    pc_value = self._get_pc_value(pc, xx, yy, xi, yi, float(rad), mode=mode)

                    if not is_right_start:
                        pc_value = 1 - pc_value

                    pcs_src[k][i] = pc_value

            # Destination
            if eq_mask[i]:
                # reuse source value when dest == src
                for k in range(len(modes)):
                    pcs_dst[k][i] = pcs_src[k][i]
            elif not dest_nan[i]:
                xdi = float(xd[i]); ydi = float(yd[i])
                for k, (mode, rad) in enumerate(zip(modes, rads)):
                    pc_value = self._get_pc_value(pc, xx, yy, xdi, ydi, float(rad), mode=mode)

                    if not is_right_start:
                        pc_value = 1 - pc_value

                    pcs_dst[k][i] = pc_value

        # Write columns
        for k, (mode, rad) in enumerate(zip(modes, rads)):
            self.events[f'pc_{mode}_src_r{rad:g}'] = pcs_src[k]
            self.events[f'pc_{mode}_dest_r{rad:g}'] = pcs_dst[k]
