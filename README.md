# rustat-python-api

### Python wrapper for the Rustat API
### Example of usage:
0. Install the package:
```bash
pip install rustat-python-api
```
1. Usage:
```python
from rustat_python_api import RuStatParser, DynamoLab, PitchControl

user = "your_login"
password = "your_password"

parser = RuStatParser(user, password)

info = parser.get_rpl_info()
keys = list(info.keys())
season_id, team_id = keys[-1], info[keys[-1]]["season_teams"][0]["id"]

schedule = parser.get_schedule(team_id, season_id)
keys = list(schedule.keys())
match_id = keys[-1]

events, subs = parser.get_events(match_id, process=True, return_subs=True)
stats = parser.get_match_stats(match_id)
tracking = parser.get_tracking(match_id)

host = "http://localhost:8001/"
client = DynamoLab(host)
client.run_model(
    model="xT",
    data=events,
    inplace=True,
    inplace_column=model
)

pc = PitchControl(tracking, events)
pc.draw_pitch_control(half=1, tp=100, save=True, filename="pitch_control")
# ffmpeg required for animation
pc.animate_pitch_control(half=1, tp=100, frames=30, filename="pitch_control")
```
