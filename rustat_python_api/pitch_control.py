import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotsoccer as mpl


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

        events.loc[:, 'pos_x'] = events.apply(
            lambda x: self.swap_coords(x, 'x'), axis=1
        )
        events.loc[:, 'pos_y'] = events.apply(
            lambda x: self.swap_coords(x, 'y'), axis=1
        )

        if ball_data is None:
            ball_data = self.interpolate_ball_data(
                events[['half', 'second', 'pos_x', 'pos_y']],
                tracking
            )

        locs_home = {
            half: {
                player_id: self.get_player_data(player_id, half, tracking)
                for player_id in tracking[tracking['side_1h'] == 'left']['player_id'].unique()
            }
            for half in tracking['half'].unique()
        }

        locs_away = {
            half: {
                player_id: self.get_player_data(player_id, half, tracking)
                for player_id in tracking[tracking['side_1h'] == 'right']['player_id'].unique()
            }
            for half in tracking['half'].unique()
        }

        locs_ball = {
            half: ball_data[ball_data['half'] == half][['pos_x', 'pos_y']].values
            for half in tracking['half'].unique()
        }

        t = {
            half: ball_data[ball_data['half'] == half]['second'].values
            for half in tracking['half'].unique()
        }

        return locs_home, locs_away, locs_ball, t


    def swap_coords(self, row, how: str = 'x'):
        half = row['half']
        team_id = row['team_id']
        possession_team_id = row['possession_team_id']
        x = row['pos_x']
        y = row['pos_y']

        if isinstance(possession_team_id, list):
            current_side = 'left' if team_id in possession_team_id else 'right'
            real_side = self.side_by_half[half][str(int(team_id))]
        else:
            current_side = 'left' if team_id == possession_team_id else 'right'
            real_side = self.side_by_half[half][str(int(team_id))]

        if current_side != real_side:
            if how == 'x':
                x = 105 - x
            else:
                y = 68 - y

        return x if how == 'x' else y

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

        # return tracking[
        #     (tracking['player_id'] == player_id)
        #     & (tracking['half'] == half)
        #     ][['pos_x', 'pos_y']].values

    def influence_function(
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

    def fit(self, half: int, tp: int, dt: int, verbose: bool = False) -> tuple:
        x = np.linspace(0, 105, dt)
        y = np.linspace(0, 68, dt)
        xx, yy = np.meshgrid(x, y)

        Zh = np.zeros(dt*dt)
        Za = np.zeros(dt*dt)

        locations = np.c_[xx.flatten(),yy.flatten()]

        for k in self.locs_home[half].keys():
            # if len(self.locs_home[half][k]) >= tp:
            Zh += self.influence_function(k, locations, tp, 'h', half, verbose)
        for k in self.locs_away[half].keys():
            # if len(self.locs_away[half][k]) >= tp:
            Za += self.influence_function(k, locations, tp, 'a', half, verbose)

        Zh = Zh.reshape((dt, dt))
        Za = Za.reshape((dt, dt))
        result = 1 / (1 + np.exp(-Za + Zh))

        return result, xx, yy

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

        fig, ax = plt.subplots(figsize=(10.5, 6.8))
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
