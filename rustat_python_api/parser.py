import requests
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from .urls import URLs


class RuStatParser:
    def __init__(self, user: str, password: str):
        self.numeric_columns = [
            'id', 'number', 'player_id', 'team_id', 'half', 'second',
            'pos_x', 'pos_y', 'pos_dest_x', 'pos_dest_y', 'len', 'possession_id', 'possession_team_id',
            'opponent_id', 'opponent_team_id', 'zone_id', 'zone_dest_id',
            'possession_number', 'attack_status_id', 'attack_team_id', 'assistant_id', 'touches', 'xg'
        ]

        self.user = user
        self.password = password

        self.cached_info = {}

    @staticmethod
    def resp2data(query: str) -> dict:
        response = requests.get(query)
        return response.json()

    def get_rpl_info(self):
        for season_id in tqdm(range(1, 36)):
            data = self.resp2data(
                URLs["tournament_teams"].format(
                    user=self.user,
                    password=self.password,
                    season_id=season_id
                )
            )

            if data:
                first_team_id = data["data"]["row"][0]["id"]
                first_team_schedule = self.resp2data(
                    URLs["schedule"].format(
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
            URLs["schedule"].format(
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

    def get_events(self, match_id: int) -> pd.DataFrame | None:
        data = self.resp2data(
            URLs["events"].format(
                user=self.user,
                password=self.password,
                match_id=match_id
            )
        )

        if not data:
            return None

        df = pd.json_normalize(data["data"]["row"])

        numeric_columns = [column for column in self.numeric_columns if column in df.columns]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        return df

    def get_match_stats(self, match_id: int) -> dict:
        data = self.resp2data(
            URLs["match_stats"].format(
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
