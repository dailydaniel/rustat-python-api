import requests
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import time

from .urls import URLs
from .config import numeric_columns
from .processing import processing


class RuStatParser:
    def __init__(
        self,
        user: str,
        password: str,
        urls: dict = URLs,
        sleep: int = -1
    ):
        self.user = user
        self.password = password
        self.urls = urls
        self.sleep = sleep

        self.cached_info = {}

    def resp2data(self, query: str) -> dict:

        if self.sleep > 0:
            time.sleep(self.sleep)

        response = requests.get(query)
        return response.json()

    def get_season_teams(self, season_id: int):
        data = self.resp2data(
            self.urls["tournament_teams"].format(
                user=self.user,
                password=self.password,
                season_id=season_id
            )
        )

        if data:
            return data['data']['row']
        else:
            return []

    def get_rpl_info(self, start_season: int = 1, end_season: int = 37):
        for season_id in tqdm(range(start_season, end_season)):
            data = self.resp2data(
                self.urls["tournament_teams"].format(
                    user=self.user,
                    password=self.password,
                    season_id=season_id
                )
            )

            if data:
                first_team_id = data["data"]["row"][0]["id"]
                first_team_schedule = self.resp2data(
                    self.urls["schedule"].format(
                        user=self.user,
                        password=self.password,
                        team_id=first_team_id,
                        season_id=season_id
                    )
                )

                if first_team_schedule:
                    last_match = first_team_schedule["data"]["row"][0]
                    season_name = f'{last_match["tournament_name"]} {last_match["season_name"]}'
                else:
                    season_name = ""

                self.cached_info[season_id] = {
                    "season_name": season_name,
                    "season_teams": data["data"]["row"]
                }

        return self.cached_info

    def get_schedule(self, team_id: str, season_id: str) -> dict:
        data = self.resp2data(
            self.urls["schedule"].format(
                user=self.user,
                password=self.password,
                team_id=team_id,
                season_id=season_id
            )
        )

        if not data:
            return {}

        return {
            int(row["id"]): {
                "match_date": row["match_date"],
                "team1_id": int(row["team1_id"]),
                "team2_id": int(row["team2_id"]),
                "team1_name": row["team1_name"],
                "team2_name": row["team2_name"],
                "round_name": (row["round_name"] if "round_name" in row else None),
                "tournament_name": (row["tournament_name"] if "tournament_name" in row else None),
                "season_name": (row["season_name"] if "season_name" in row else None)
            }
            for row in data["data"]["row"]
        }

    def get_events(
        self,
        match_id: int,
        process: bool = True,
        return_subs: bool = True
    ) -> pd.DataFrame | None | tuple[pd.DataFrame, pd.DataFrame]:
        data = self.resp2data(
            self.urls["events"].format(
                user=self.user,
                password=self.password,
                match_id=match_id
            )
        )

        if not data:
            return None

        df = pd.json_normalize(data["data"]["row"])

        current_numeric_columns = [column for column in numeric_columns if column in df.columns]
        df[current_numeric_columns] = df[current_numeric_columns].apply(pd.to_numeric, errors='coerce')

        if process:
            df['match_id'] = match_id

            if return_subs:
                subs = df[df['action_id'] == '14000'][[
                    'match_id', 'half', 'second',
                    'team_id', 'team_name',
                    'opponent_id', 'opponent_name',
                    'player_id', 'player_name'
                ]].rename(columns={
                    'player_id': 'player_id_out',
                    'opponent_id': 'player_id_in',
                    'player_name': 'player_name_out',
                    'opponent_name': 'player_name_in'
                })

            df = processing(df)

        return (df, subs) if return_subs else df

    def get_tracking(self, match_id: int) -> pd.DataFrame | None:
        data = self.resp2data(
            self.urls["tracking"].format(
                user=self.user,
                password=self.password,
                match_id=match_id
            )
        )

        if not data:
            return None

        data = data["data"]["team"]
        df = pd.DataFrame(columns=["half", "second", "pos_x", "pos_y", "team_id", "player_id", "player_name", "side_1h"])

        for team_data in tqdm(data):
            team_id = team_data["id"]
            side_1h = team_data["gate_position_half_1"]

            for player_data in team_data["player"]:
                player_id = player_data["id"]
                player_name = player_data["name"]

                cur_df = pd.json_normalize(player_data["row"])
                cur_df = cur_df.apply(pd.to_numeric, errors='coerce')
                cur_df["team_id"] = team_id
                cur_df["player_id"] = player_id
                cur_df["player_name"] = player_name
                cur_df["side_1h"] = side_1h

                df = pd.concat([df, cur_df], ignore_index=True)

        df = df.sort_values(by=['half', 'second', 'team_id', 'player_id']).reset_index(drop=True)
        df['pos_x'] = df['pos_x'] + 105/2
        df['second'] = df['second'].astype(int)

        return df

    def get_tracking_30fps(
        self,
        match_id: int, half: int, lang_id: int = 1,
        referee_data: int = 0, ball_data: int = 1
    ) -> pd.DataFrame | tuple | None:
        data = self.resp2data(
            self.urls["tracking_30fps"].format(
                user=self.user,
                password=self.password,
                match_id=match_id,
                half=half,
                lang_id=lang_id,
                referee_data=referee_data,
                ball_data=ball_data
            )
        )

        if not data:
            return None

        teams_data = data['data']['teams']
        if 'ball' in data['data']:
            ball_data = data['data']['ball']
        if 'referee' in data['data']:
            referee_data = data['data']['referee']
        del data

        tracking = pd.DataFrame(columns=[
            "half", "second", "pos_x", "pos_y", "team_id", "player_id", "player_name", "distance", "speed"
        ])

        for team_data in tqdm(teams_data):
            team_id = team_data["id"]
            side_1h = team_data["gate_position_half_1"]

            for player_data in team_data["players"]:
                player_id = player_data["id"]
                player_name = player_data["name"]

                cur_df = pd.json_normalize(player_data["rows"])
                cur_df = cur_df.apply(pd.to_numeric, errors='coerce')
                cur_df["team_id"] = team_id
                cur_df["player_id"] = player_id
                cur_df["player_name"] = player_name
                cur_df["side_1h"] = side_1h

                tracking = pd.concat([tracking, cur_df], ignore_index=True)

        tracking['second'] = tracking['second'].astype(float)
        tracking = tracking.sort_values(by=['half', 'second', 'team_id', 'player_id']).reset_index(drop=True)
        tracking['pos_x'] = tracking['pos_x'] + 105/2
        tracking['team_id'] = tracking['team_id'].astype(str)

        if referee_data:
            referee = pd.json_normalize(referee_data['rows'])
            referee['second'] = referee['second'].astype(float)
            referee = referee.sort_values(by=['half', 'second']).reset_index(drop=True)
            referee['pos_x'] = referee['pos_x'] + 105/2

        if ball_data:
            ball = pd.json_normalize(ball_data['rows'])
            ball['second'] = ball['second'].astype(float)
            ball = ball.sort_values(by=['half', 'second']).reset_index(drop=True)
            ball['pos_x'] = ball['pos_x'] + 105/2

        if referee_data and ball_data:
            return tracking, referee, ball
        elif referee_data:
            return tracking, referee
        elif ball_data:
            return tracking, ball
        else:
            return tracking

    def get_match_stats(self, match_id: int) -> dict:
        data = self.resp2data(
            self.urls["match_stats"].format(
                user=self.user,
                password=self.password,
                match_id=match_id
            )
        )

        if not data:
            return {}

        stats = defaultdict(dict)

        for row in data['data']['row']:
            team_id = int(row['team_id'])
            param_name = row['param_name']

            param_value = float(row['value'])

            stats[param_name][team_id] = param_value

        return stats

    def get_players_match_stats(self, match_id: int) -> dict:
        data = self.resp2data(
            self.urls["player_match_stats"].format(
                user=self.user,
                password=self.password,
                match_id=match_id
            )
        )

        if not data:
            return {}

        return data['data']['team']

    def get_players_minutes_in_match(self, match_id: int) -> dict:
        data = self.get_players_match_stats(match_id)

        if not data:
            return {}

        players_minutes = {}

        for team_data in data:
            for player_data in team_data['player']:
                player_id = int(player_data['id'])

                minutes = [float(metric['value']) for metric in player_data['param'] if metric['id'] == '288']
                minutes = minutes[0] if minutes else 0

                players_minutes[player_id] = minutes

        return players_minutes
