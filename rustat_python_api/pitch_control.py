import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotsoccer as mpl
import torch
from tqdm import tqdm

try:
    from .kernels import triton_influence
except ImportError:
    triton_influence = None

class PitchControl:
    def __init__(self, tracking: pd.DataFrame, events: pd.DataFrame, ball_data: pd.DataFrame = None):
        self.team_ids = tracking['team_id'].unique()
        sides = tracking.groupby('team_id')['side_1h'].unique()
        side_by_team = dict(zip(self.team_ids, sides[self.team_ids].apply(lambda x: x[0])))
        self.side_by_half = {
            1: side_by_team,
            2:
                {
                    team: 'left' if side == 'right' else 'right'
                    for team, side in side_by_team.items()
                }
        }

        self._grid_cache: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self.locs_home, self.locs_away, self.locs_ball, self.t = self.get_locs(
            tracking,
            events,
            ball_data
        )

    def get_locs(self, tracking: pd.DataFrame, events: pd.DataFrame, ball_data: pd.DataFrame | None) -> tuple:
        events = events[[
            'possession_number', 'team_id', 'possession_team_id',
            'half', 'second', 'pos_x', 'pos_y'
        ]]

        events = self.swap_coords_batch(events)

        if ball_data is None:
            ball_data = self.interpolate_ball_data(
                events[['half', 'second', 'pos_x', 'pos_y']],
                tracking
            )

        locs_home, locs_away = self.build_player_locs(tracking)

        locs_ball = {
            half: ball_data[ball_data['half'] == half][['pos_x', 'pos_y']].values
            for half in tracking['half'].unique()
        }

        t = {
            half: ball_data[ball_data['half'] == half]['second'].values
            for half in tracking['half'].unique()
        }

        return locs_home, locs_away, locs_ball, t

    # def swap_coords(self, row, how: str = 'x'):
    #     half = row['half']
    #     team_id = row['team_id']
    #     possession_team_id = row['possession_team_id']
    #     x = row['pos_x']
    #     y = row['pos_y']

    #     if isinstance(possession_team_id, list):
    #         current_side = 'left' if team_id in possession_team_id else 'right'
    #         real_side = self.side_by_half[half][str(int(team_id))]
    #     else:
    #         current_side = 'left' if team_id == possession_team_id else 'right'
    #         real_side = self.side_by_half[half][str(int(team_id))]

    #     if current_side != real_side:
    #         if how == 'x':
    #             x = 105 - x
    #         else:
    #             y = 68 - y

    #     return x if how == 'x' else y

    def swap_coords_batch(self, events: pd.DataFrame) -> pd.DataFrame:
        """Vectorised replacement for per-row `swap_coords`.

        Modifies *events* in-place: flips coordinates for rows where the
        current attacking direction (left/right) does not match the
        canonical side stored in ``self.side_by_half``.
        Returns the same DataFrame for chaining.
        """

        side_by_half = self.side_by_half  # local alias for speed

        def needs_swap(row):
            team_id = row['team_id']
            poss = row['possession_team_id']
            half = row['half']

            current_left = team_id in poss if isinstance(poss, list) else team_id == poss
            current_side = 'left' if current_left else 'right'
            real_side = side_by_half[half][str(int(team_id))]
            return current_side != real_side

        mask = events.apply(needs_swap, axis=1)

        # flip coords in bulk
        events.loc[mask, 'pos_x'] = 105 - events.loc[mask, 'pos_x']
        events.loc[mask, 'pos_y'] = 68  - events.loc[mask, 'pos_y']
        return events

    def build_player_locs(self, tracking: pd.DataFrame):
        """Vectorised construction of player location dictionaries.

        Returns (locs_home, locs_away) where each is
        {half: {player_id: np.ndarray(T,2)}}.
        """
        locs_home = {1: {}, 2: {}}
        locs_away = {1: {}, 2: {}}

        # Work per half to keep order and avoid extra boolean checks.
        for half in (1, 2):
            half_df = tracking[tracking['half'] == half]
            for side, locs_out in [('left', locs_home), ('right', locs_away)]:
                side_df = half_df[half_df['side_1h'] == side]
                for pid, grp in side_df.groupby('player_id'):
                    locs_out[half][pid] = grp[['pos_x', 'pos_y']].values
        return locs_home, locs_away

    @staticmethod
    def interpolate_ball_data(
        ball_data: pd.DataFrame,
        player_data: pd.DataFrame
    ) -> pd.DataFrame:
        ball_data = ball_data.drop_duplicates(subset=['second', 'half'])

        interpolated_data = []
        for half in ball_data['half'].unique():
            ball_half = ball_data[ball_data['half'] == half]
            player_half = player_data[player_data['half'] == half]

            player_times = player_half['second'].unique()

            ball_half = ball_half.sort_values(by='second')
            interpolated_half = pd.DataFrame({'second': player_times})
            interpolated_half['pos_x'] = np.interp(
                interpolated_half['second'], ball_half['second'], ball_half['pos_x']
            )
            interpolated_half['pos_y'] = np.interp(
                interpolated_half['second'], ball_half['second'], ball_half['pos_y']
            )
            interpolated_half['half'] = half
            interpolated_data.append(interpolated_half)

        interpolated_ball_data = pd.concat(interpolated_data, ignore_index=True)
        return interpolated_ball_data

    @staticmethod
    def get_player_data(player_id, half, tracking):
        timestamps = tracking[tracking['half'] == half]['second'].unique()
        player_data = tracking[
            (tracking['player_id'] == player_id)
            & (tracking['half'] == half)
        ][['second', 'pos_x', 'pos_y']]

        player_data_full = pd.DataFrame({'second': timestamps})
        player_data_full = player_data_full.merge(player_data, on='second', how='left')

        return player_data_full[['pos_x', 'pos_y']].values

    def influence_np(
        self,
        player_index: str,
        location: np.ndarray,
        time_index: int,
        home_or_away: str,
        half: int,
        verbose: bool = False
    ):
        if home_or_away == 'h':
            data = self.locs_home[half].copy()
        elif home_or_away == 'a':
            data = self.locs_away[half].copy()
        else:
            raise ValueError("Enter either 'h' or 'a'.")

        locs_ball = self.locs_ball[half].copy()
        t = self.t[half].copy()

        if (
                np.all(np.isfinite(data[player_index][[time_index, time_index + 1], :]))
                & np.all(np.isfinite(locs_ball[time_index, :]))
        ):
            jitter = 1e-10 ## to prevent identically zero covariance matrices when velocity is zero
            ## compute velocity by fwd difference
            s = (
                    np.linalg.norm(
                        data[player_index][time_index + 1,:]
                        - data[player_index][time_index,:] + jitter
                    )
                    / (t[time_index + 1] - t[time_index])
            )
            ## velocities in x,y directions
            sxy = (
                    (data[player_index][time_index + 1, :] - data[player_index][time_index, :] + jitter)
                    / (t[time_index + 1] - t[time_index])
            )
            ## angle between velocity vector & x-axis
            theta = np.arccos(sxy[0] / np.linalg.norm(sxy))
            ## rotation matrix
            R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            mu = data[player_index][time_index, :] + sxy * 0.5
            Srat = (s / 13) ** 2
            Ri = np.linalg.norm(locs_ball[time_index, :] - data[player_index][time_index, :])
            ## don't think this function is specified in the paper but looks close enough to fig 9
            Ri = np.minimum(4 + Ri ** 3/ (18 ** 3 / 6), 10)
            S = np.array([[(1 + Srat) * Ri / 2, 0], [0, (1 - Srat) * Ri / 2]])
            Sigma = np.matmul(R, S)
            Sigma = np.matmul(Sigma, S)
            Sigma = np.matmul(Sigma, np.linalg.inv(R)) ## this is not efficient, forgive me.
            out = mvn.pdf(location, mu, Sigma) / mvn.pdf(data[player_index][time_index, :], mu, Sigma)
        else:
            if verbose:
                print("Data is not finite.")
            out = np.zeros(location.shape[0])
        return out

    def _batch_influence_pt(
        self,
        player_dict: dict,
        locs: torch.Tensor,
        time_index: int,
        half: int,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute cumulative influence of *many* players at once.

        Parameters
        ----------
        player_dict : dict[player_id -> np.ndarray(shape=(T,2))]
            Pre-loaded trajectory arrays for one team.
        locs : torch.Tensor shape (N,2)
            Grid locations (already on correct device / dtype).
        time_index : int
            Frame index t.
        half : int
            Half number.
        device, dtype : torch configuration.

        Returns
        -------
        torch.Tensor shape (N,)
            Sum of influences from all valid players in *player_dict*.
        """

        pos_t_list, pos_tp1_list = [], []
        for arr in player_dict.values():
            # Ensure we have t and t+1 and no NaNs at those rows
            if (
                time_index + 1 < arr.shape[0]
                and np.isfinite(arr[[time_index, time_index + 1], :]).all()
            ):
                pos_t_list.append(arr[time_index])
                pos_tp1_list.append(arr[time_index + 1])

        if not pos_t_list:
            return torch.zeros(locs.shape[0], device=device, dtype=dtype)

        pos_t = torch.tensor(np.asarray(pos_t_list), device=device, dtype=dtype)  # (P,2)
        pos_tp1 = torch.tensor(np.asarray(pos_tp1_list), device=device, dtype=dtype)  # (P,2)

        # Velocity, speed, rotation -------------------------
        dt_sec = float(self.t[half][time_index + 1] - self.t[half][time_index])
        sxy = (pos_tp1 - pos_t) / dt_sec  # (P,2)

        speed = torch.linalg.norm(sxy, dim=1)  # (P,)
        norm_sxy = speed.clamp(min=1e-6)

        theta = torch.acos(torch.clamp(sxy[:, 0] / norm_sxy, -1 + 1e-6, 1 - 1e-6))  # (P,)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        R = torch.stack(
            [
                torch.stack([cos_t, -sin_t], dim=1),
                torch.stack([sin_t,  cos_t], dim=1),
            ],
            dim=1,
        )  # (P,2,2)

        # Shape parameters ----------------------------------
        Srat = (speed / 13) ** 2  # (P,)

        ball_pos = torch.tensor(
            self.locs_ball[half][time_index], device=device, dtype=dtype
        )  # (2,)
        Ri = torch.linalg.norm(ball_pos - pos_t, dim=1)  # (P,)
        Ri = torch.minimum(4 + Ri ** 3 / (18 ** 3 / 6), torch.tensor(10.0, device=device, dtype=dtype))

        S11 = (1 + Srat) * Ri / 2
        S22 = (1 - Srat) * Ri / 2

        S = torch.zeros((pos_t.shape[0], 2, 2), device=device, dtype=dtype)
        S[:, 0, 0] = S11
        S[:, 1, 1] = S22

        Sigma = R @ S @ S @ R.transpose(1, 2)  # (P,2,2)

        eye = torch.eye(2, device=device, dtype=dtype) * 1e-6
        eye = eye.expand(pos_t.shape[0], 2, 2)  # broadcast to (P,2,2)

        if dtype == torch.float16:
            Sigma_inv = torch.linalg.inv((Sigma + eye).float()).to(dtype)
        else:
            Sigma_inv = torch.linalg.inv(Sigma + eye)

        # Mean ----------------------------------------------
        mu = pos_t + 0.5 * sxy  # (P,2)

        # Grid diff & Mahalanobis ----------------------------
        diff = locs.view(1, -1, 2)  # (1,N,2)
        diff = diff - mu.unsqueeze(1)   # (P,N,2)

        device = torch.device(device)

        if device.type == 'cuda':
            out = triton_influence(
                mu.unsqueeze(0), Sigma_inv.unsqueeze(0),
                locs, BLOCK_N=64
            )[0]  # (N,)

            return out
        else:
            maha = torch.einsum('pni,pij,pnj->pn', diff, Sigma_inv, diff)  # (P,N)
            maha = torch.nan_to_num(maha, nan=1e9, posinf=1e9, neginf=1e9)
            out = torch.exp(-0.5 * maha)  # (P,N)

            return out.sum(dim=0)  # sum over players

    def _batch_team_influence_frames_pt(
        self,
        pos_t: torch.Tensor,        # (F,P,2)
        pos_tp1: torch.Tensor,      # (F,P,2)
        ball_pos: torch.Tensor,     # (F,2)
        dt_secs: torch.Tensor,      # (F,)
        locs: torch.Tensor,         # (N,2)
        dtype: torch.dtype,
    ) -> torch.Tensor:              # returns (F,N)
        """Vectorised influence for many frames & players of ОДНОЙ команды."""

        device = locs.device

        sxy = (pos_tp1 - pos_t) / dt_secs[:, None, None]  # (F,P,2)
        speed = torch.linalg.norm(sxy, dim=-1)            # (F,P)
        norm_sxy = speed.clamp(min=1e-6)

        theta = torch.acos(torch.clamp(sxy[..., 0] / norm_sxy, -1 + 1e-6, 1 - 1e-6))
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)  # (F,P)

        R = torch.stack(
            [
                torch.stack([cos_t, -sin_t], dim=-1),
                torch.stack([sin_t,  cos_t], dim=-1),
            ],
            dim=-2,
        )  # (F,P,2,2)

        Srat = (speed / 13) ** 2  # (F,P)

        Ri = torch.linalg.norm(ball_pos[:, None, :] - pos_t, dim=-1)  # (F,P)
        Ri = torch.minimum(4 + Ri ** 3 / (18 ** 3 / 6), torch.tensor(10.0, device=device, dtype=dtype))

        S11 = (1 + Srat) * Ri / 2
        S22 = (1 - Srat) * Ri / 2

        S = torch.zeros((*pos_t.shape[:-1], 2, 2), device=device, dtype=dtype)  # (F,P,2,2)
        S[..., 0, 0] = S11
        S[..., 1, 1] = S22

        Sigma = R @ S @ S @ R.transpose(-1, -2)  # (F,P,2,2)

        eye = torch.eye(2, device=device, dtype=dtype) * 1e-6
        eye = eye.view(1, 1, 2, 2)

        if dtype == torch.float16:
            Sigma_inv = torch.linalg.inv((Sigma + eye).float()).to(dtype)
        else:
            Sigma_inv = torch.linalg.inv(Sigma + eye)

        mu = pos_t + 0.5 * sxy  # (F,P,2)

        diff = locs.view(1, 1, -1, 2)  # (1,1,N,2)
        diff = diff - mu.unsqueeze(2)   # (F,P,N,2)

        if device.type == 'cuda':
            out = triton_influence(mu, Sigma_inv, locs, BLOCK_N=64)  # (F,N)

            return out
        else:
            maha = torch.einsum('fpni,fpij,fpnj->fpn', diff, Sigma_inv, diff)  # (F,P,N)
            maha = torch.nan_to_num(maha, nan=1e9, posinf=1e9, neginf=1e9)
            out = torch.exp(-0.5 * maha)  # (F,P,N)

            return out.sum(dim=1)  # sum over players

    @staticmethod
    def _stack_team_frames(players: list[np.ndarray], frames: np.ndarray, device: str, dtype: torch.dtype):
        """Stack positions for given frames into torch tensors (pos_t, pos_tp1)."""
        # Ensure every player's trajectory is long enough; if not, pad by repeating
        # the last available coordinate so that indexing `frames` and `frames+1` is safe.
        max_needed = frames[-1] + 1  # we access idx and idx+1

        padded = []
        for p in players:
            if len(p) <= max_needed:
                pad_len = max_needed + 1 - len(p)
                if pad_len > 0:
                    last = p[-1][None, :]
                    p = np.vstack([p, np.repeat(last, pad_len, axis=0)])
            padded.append(p)

        pos_t = torch.tensor(
            np.stack([p[frames] for p in padded], axis=1), device=device, dtype=dtype
        )  # (F,P,2)
        pos_tp1 = torch.tensor(
            np.stack([p[frames + 1] for p in padded], axis=1), device=device, dtype=dtype
        )
        return pos_t, pos_tp1

    def _fit_full_pt(
        self,
        half: int,
        dt: int,
        device: str,
        batch_size: int,
        use_fp16: bool,
        verbose: bool,
    ):
        """Internal helper with fully batched PyTorch implementation."""

        dtype = torch.float16 if use_fp16 else torch.float32
        xx_t, yy_t, locs_t = self._get_grid(dt, device, dtype)
        xx, yy = xx_t.cpu().numpy(), yy_t.cpu().numpy()

        T = len(self.t[half]) - 1
        pc_all = np.empty((T, dt, dt), dtype=np.float32)

        home_players = list(self.locs_home[half].values())
        away_players = list(self.locs_away[half].values())

        for start in tqdm(range(0, T, batch_size)):
            end = min(start + batch_size, T)
            frames = np.arange(start, end)

            # deltas t
            dt_secs = torch.tensor(
                self.t[half][frames + 1] - self.t[half][frames],
                device=device,
                dtype=dtype,
            )

            ball_pos = torch.tensor(
                self.locs_ball[half][frames], device=device, dtype=dtype
            )

            # stack teams
            pos_t_h, pos_tp1_h = self._stack_team_frames(home_players, frames, device, dtype)
            pos_t_a, pos_tp1_a = self._stack_team_frames(away_players, frames, device, dtype)

            Zh = self._batch_team_influence_frames_pt(
                pos_t_h, pos_tp1_h, ball_pos, dt_secs, locs_t, dtype
            )
            Za = self._batch_team_influence_frames_pt(
                pos_t_a, pos_tp1_a, ball_pos, dt_secs, locs_t, dtype
            )

            res_batch = torch.sigmoid(Za - Zh).reshape(-1, dt, dt)
            pc_all[start:end] = res_batch.cpu().numpy().astype(np.float32)

            if verbose:
                print(f"pt full: frames {start}-{end-1} done")

        return pc_all, xx, yy

    def _get_grid(self, dt: int, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper to create a grid of locations for torch-based pitch control."""
        x = torch.linspace(0, 105, dt, device=device, dtype=dtype)
        y = torch.linspace(0, 68, dt, device=device, dtype=dtype)
        xx_t, yy_t = torch.meshgrid(x, y, indexing="xy")  # (dt,dt)
        locs_t = torch.stack([xx_t, yy_t], dim=-1).reshape(-1, 2)  # (N,2)
        return xx_t, yy_t, locs_t

    def _fit_pt(
        self,
        half: int,
        tp: int,
        dt: int = 200,
        device: str = "cpu",
        verbose: bool = False,
        use_fp16: bool = True,
    ):
        """Torch-accelerated pitch-control calculation.

        Returns
        -------
        result : np.ndarray shape (dt,dt)
        xx, yy : np.ndarray meshgrids (dt,dt)
        """
        if torch is None:
            raise ImportError("PyTorch is required for backend='torch'.")

        dtype = torch.float16 if (use_fp16 and device != "cpu") else torch.float32

        # ---- grid caching ----
        key = (dt, device, dtype)
        if key not in self._grid_cache:
            xx_t, yy_t, locs_t = self._get_grid(dt, device, dtype)
            # Store; keep cache small (max 3 grids)
            if len(self._grid_cache) >= 3:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[key] = (xx_t, yy_t, locs_t)
        else:
            xx_t, yy_t, locs_t = self._grid_cache[key]

        # --- vectorised influence computation ---
        Zh = self._batch_influence_pt(
            self.locs_home[half], locs_t, tp, half, device, dtype
        )
        Za = self._batch_influence_pt(
            self.locs_away[half], locs_t, tp, half, device, dtype
        )

        res_t = torch.sigmoid(Za - Zh).reshape(dt, dt)

        # Convert to numpy for downstream plotting
        return res_t.cpu().numpy(), xx_t.cpu().numpy(), yy_t.cpu().numpy()

    def _fit_np(self, half: int, tp: int, dt: int, verbose: bool = False) -> tuple:
        x = np.linspace(0, 105, dt)
        y = np.linspace(0, 68, dt)
        xx, yy = np.meshgrid(x, y)

        Zh = np.zeros(dt*dt)
        Za = np.zeros(dt*dt)

        locations = np.c_[xx.flatten(),yy.flatten()]

        for k in self.locs_home[half].keys():
            # if len(self.locs_home[half][k]) >= tp:
            Zh += self.influence_np(k, locations, tp, 'h', half, verbose)
        for k in self.locs_away[half].keys():
            # if len(self.locs_away[half][k]) >= tp:
            Za += self.influence_np(k, locations, tp, 'a', half, verbose)

        Zh = Zh.reshape((dt, dt))
        Za = Za.reshape((dt, dt))
        result = 1 / (1 + np.exp(-Za + Zh))

        return result, xx, yy

    def fit(
        self,
        half: int,
        tp: int,
        dt: int = 100,
        backend: str = "np",
        device: str = "cpu",
        verbose: bool = False,
        use_fp16: bool = True,
    ):
        """Selects NumPy or PyTorch backend depending on `backend`."""
        match backend:
            case "np" | "numpy":
                return self._fit_np(half, tp, dt, verbose)
            case "torch" | "pt":
                return self._fit_pt(half, tp, dt, device=device, verbose=verbose, use_fp16=use_fp16)
            case _:
                raise ValueError(f"Unknown backend '{backend}'. Use 'np' or 'torch'.")

    def fit_full(
        self,
        half: int,
        dt: int = 100,
        backend: str = "np",
        device: str = "cpu",
        batch_size: int = 30*60,
        use_fp16: bool = True,
        verbose: bool = False,
    ):
        """Compute pitch-control map for *каждый* кадр тайма.

        Returns
        -------
        maps : np.ndarray, shape (T, dt, dt)
            Pitch-control probability for home team at every frame.
        xx, yy : np.ndarray, shape (dt, dt)
            Coordinate grids (общие для всех кадров).
        """

        T = len(self.t[half]) - 1  # мы используем t и t+1, поэтому последний кадр T-1

        match backend:
            case "np" | "numpy":
                pc_all = np.empty((T, dt, dt), dtype=np.float32)
                for tp in tqdm(range(T)):
                    pc_map, xx, yy = self._fit_np(half, tp, dt, verbose=False)
                    pc_all[tp] = pc_map.astype(np.float32)
                    if verbose and tp % 500 == 0:
                        print(f"np full-match: done {tp}/{T}")
                return pc_all, xx, yy

            case "torch" | "pt":
                return self._fit_full_pt(
                    half, dt, device, batch_size, use_fp16, verbose
                )
            case _:
                raise ValueError("backend must be 'np' or 'pt'")

    def draw_pitch_control(
        self,
        half: int,
        tp: int,
        pitch_control: tuple = None,
        save: bool = False,
        dt: int = 200,
        filename: str = 'pitch_control'
    ):
        if pitch_control is None:
            pitch_control, xx, yy = self.fit(half, tp, dt)
        else:
            pitch_control, xx, yy = pitch_control

        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        # mpl.field(fieldcolor="white", linecolor="black", alpha=1, show=False, ax=ax)
        mpl.field("white", show=False, ax=ax)

        plt.contourf(xx, yy, pitch_control)

        for k in self.locs_home[half].keys():
            # if len(self.locs_home[half][k]) >= tp:
            if np.isfinite(self.locs_home[half][k][tp, :]).all():
                plt.scatter(
                    self.locs_home[half][k][tp, 0],
                    self.locs_home[half][k][tp, 1],
                    color='darkgrey'
                )

        for k in self.locs_away[half].keys():
            # if len(self.locs_away[half][k]) >= tp:
            if np.isfinite(self.locs_away[half][k][tp, :]).all():
                plt.scatter(
                    self.locs_away[half][k][tp, 0],
                    self.locs_away[half][k][tp, 1], color='black'
                )

        plt.scatter(
            self.locs_ball[half][tp, 0],
            self.locs_ball[half][tp, 1],
            color='red'
        )

        if save:
            plt.savefig(f'{filename}.png', dpi=300)
        else:
            plt.show()

    def animate_pitch_control(
        self,
        half: int,
        tp: int,
        filename: str = "pitch_control_animation",
        dt: int = 200,
        frames: int = 30,
        interval: int = 1000
    ):
        """
        ffmpeg should be installed on your machine.
        """
        fig, ax = plt.subplots(figsize=(10.5, 6.8))

        def animate(i):
            fr = tp + i
            pitch_control, xx, yy = self.fit(half, fr, dt)

            mpl.field("white", show=False, ax=ax)
            ax.axis('off')

            plt.contourf(xx, yy, pitch_control)

            for k in self.locs_home[half].keys():
                # if len(self.locs_home[half][k]) >= fr:
                if np.isfinite(self.locs_home[half][k][fr, :]).all():
                    plt.scatter(
                        self.locs_home[half][k][fr, 0],
                        self.locs_home[half][k][fr, 1],
                        color='darkgrey'
                    )
            for k in self.locs_away[half].keys():
                # if len(self.locs_away[half][k]) >= fr:
                if np.isfinite(self.locs_away[half][k][fr, :]).all():
                    plt.scatter(
                        self.locs_away[half][k][fr, 0],
                        self.locs_away[half][k][fr, 1],
                        color='black'
                    )

            plt.scatter(
                self.locs_ball[half][fr, 0],
                self.locs_ball[half][fr, 1],
                color='red'
            )

            return ax

        x = np.linspace(0, 105, dt)
        y = np.linspace(0, 68, dt)
        xx, yy = np.meshgrid(x, y)

        ani = animation.FuncAnimation(
            fig=fig,
            func=animate,
            frames=min(frames, len(self.locs_ball[half]) - tp),
            interval=interval,
            blit=False
        )

        ani.save(f'{filename}.mp4', writer='ffmpeg')
