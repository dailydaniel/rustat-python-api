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

    def get_rpl_info(self):
        for season_id in tqdm(range(1, 36)):
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
                "team2_name": row["team2_name"]
            }
            for row in data["data"]["row"]
        }

    def get_events(self, match_id: int, process: bool = True) -> pd.DataFrame | None:
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
            df = processing(df)

        return df

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

        return df.sort_values(by=["second", "team_id", "player_id"])

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
