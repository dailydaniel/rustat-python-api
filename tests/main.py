import os

from dotenv import load_dotenv

import sys
sys.path.append('../')

from rustat_python_api import RuStatParser


def test_info(user: str, password: str, sleep: float = 0.25):
    parser = RuStatParser(user, password, sleep=sleep)
    info = parser.get_rpl_info()
    keys = list(info.keys())
    print(keys)
    print(info[keys[-1]]["season_name"])
    print(info[keys[-1]]["season_teams"])

    return keys[-1], info[keys[-1]]["season_teams"][0]["id"]


def test_schedule(user: str, password: str, team_id: str, season_id: str):
    parser = RuStatParser(user, password)
    schedule = parser.get_schedule(team_id, season_id)
    keys = list(schedule.keys())
    print(keys)
    print(schedule[keys[-1]])

    return keys[-1]

def test_events(user: str, password: str, match_id: str):
    parser = RuStatParser(user, password)
    events = parser.get_events(match_id)

    print(events.describe())

def test_stats(user: str, password: str, match_id: str):
    parser = RuStatParser(user, password)
    stats = parser.get_match_stats(match_id)

    keys = list(stats.keys())
    print(keys)
    print(stats[keys[-1]])

def test_tracking(user: str, password: str, match_id: str):
    parser = RuStatParser(user, password)
    tracking = parser.get_tracking(match_id)

    print(tracking.describe())
    print(tracking["player_name"].unique())


if __name__ == "__main__":
    load_dotenv(dotenv_path='.env')

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    sleep = 0.25

    print(user, password)

    season_id, team_id = test_info(user, password, sleep=sleep)
    match_id = test_schedule(user, password, team_id, season_id)
    test_events(user, password, match_id)
    test_stats(user, password, match_id)
    test_tracking(user, password, match_id)
