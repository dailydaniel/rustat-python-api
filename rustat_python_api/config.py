columns = [
    'player_name', 'team_name', 'half', 'second', 'action_id', 'action_name',
    'position_name', 'possession_number', 'pos_x', 'pos_y', 'pos_dest_x', 'pos_dest_y',
    'player_id', 'number', 'team_id', 'standart_name', 'possession_time',
    'opponent_id', 'opponent_name', 'opponent_team_id', 'opponent_team_name',
    'opponent_position_name', 'zone_name', 'zone_dest_name', 'len',
    'possession_team_id', 'possession_team_name', 'possession_name',
    'attack_status_name', 'attack_type_name', 'attack_flang_name',
    'attack_team_id', 'attack_team_name', 'attack_number',
    'body_name', 'gate_x', 'gate_y', 'assistant_id',
    'assistant_name', 'shot_type', 'touches', 'xg',
    'shot_handling', 'match_id', 'receiver_id', 'receiver_name'
]

numeric_columns = [
    'id', 'number', 'player_id', 'team_id', 'half', 'second',
    'pos_x', 'pos_y', 'pos_dest_x', 'pos_dest_y', 'len', 'possession_id', 'possession_team_id',
    'opponent_id', 'opponent_team_id', 'zone_id', 'zone_dest_id',
    'possession_number', 'attack_status_id', 'attack_team_id', 'assistant_id', 'touches', 'xg'
]

id2type = {
    1: 'pass', 2: 'duel', 3: 'foul',
    4: 'shot', 5: 'free kick', 6: 'interception',
    7: 'rebound', 8: 'goal', 9: 'clearance',
    10: 'bad ball control', 11: 'control', 12: 'attack',
    13: 'keeper', 14: 'substitution', 15: 'formation',
    16: 'player position', 17: 'ball off', 18: 'match status',
    19: 'mistake', 20: 'translation problem', 21: 'carry',
    22: 'receive', 23: 'goal attack involvement', 24: 'rating',
    25: 'average position', 26: 'cross', 27: 'ball out',
    28: 'other', 29: 'video', 30: 'bad mistake',
    31: 'bad keeper mistake', 32: 'goal moment', 33: 'team pressing',
    34: 'line up', 35: 'sync', 36: 'referee',
    37: 'insurance', 38: 'injury',
    128: 'staff', 161: 'sub player'
}
