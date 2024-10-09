# rustat-python-api

### Python wrapper for the Rustat API
### Example of usage:
0. Install the package:
```bash
pip install rustat-python-api
```
1. Usage:
```python
from rustat_python_api import RuStatParser

user = "your_login"
password = "your_password"

parser = RuStatParser(user, password)

info = parser.get_rpl_info()
keys = list(info.keys())
season_id, team_id = keys[-1], info[keys[-1]]["season_teams"][0]["id"]

schedule = parser.get_schedule(team_id, season_id)
keys = list(schedule.keys())
match_id = keys[-1]

events = parser.get_events(match_id)

stats = parser.get_match_stats(match_id)
```
