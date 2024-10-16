import pandas as pd
import numpy as np


def process_list(x: pd.Series):
    lst = x.dropna().unique().tolist()
    # return str(lst)
    if len(lst) == 1:
        return lst[0]
    elif len(lst) == 0:
        return np.nan
    else:
        return lst


def gluing(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['player_id', 'half', 'second', 'pos_x', 'pos_y']

    df_gb = df.groupby(cols).agg(process_list).reset_index()
    df_gb['possession_number'] = df_gb['possession_number'].apply(
        lambda x: max(x) if isinstance(x, list) else x
    )
    df_gb = df_gb.sort_values(by=['half', 'second', 'possession_number']).reset_index(drop=True)
    return df_gb


def add_reciever(glued_df: pd.DataFrame) -> pd.DataFrame:
    df = glued_df.copy()
    df['receiver_id'] = df['player_id'].shift(1)
    df['receiver_name'] = df['player_name'].shift(1)

    mask = (
            (df['action_name'] == 'Ball receiving')
            & (df['pos_x'] == df['pos_dest_x'].shift(1))
            & (df['pos_y'] == df['pos_dest_y'].shift(1))
            & (df['team_id'] == df['team_id'].shift(1))
            & (df['player_id'] != df['player_id'].shift(1))
            & (df['possession_number'] == df['possession_number'].shift(1))
    )

    idx = df[mask].index
    remaining_idx = df.drop(idx-1).index

    df.loc[remaining_idx, 'receiver_id'] = np.nan
    df.loc[remaining_idx, 'receiver_name'] = np.nan

    df = df[df['action_name'] != 'Ball receiving'].reset_index(drop=True)

    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'player_name', 'team_name', 'half', 'second', 'action_name',
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

    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

    return df[(~df['possession_number'].isna()) | (df['second'] != 0)][columns].reset_index(drop=True)


def tag2type(tags: list[str]) -> str:
    tags = [tag.lower() for tag in tags]
    tags_str = ', '.join(tags)

    if 'pass' in tags_str or 'assist' in tags_str:
        pass_tags = [tag for tag in tags if 'pass' in tag and tag != 'pass interception']
        assist_tags = [tag for tag in tags if 'assist' in tag]
        cross_tags = [tag for tag in tags if 'cross' in tag and tag != 'cross interception']

        if len(pass_tags) > 0 or (len(assist_tags) > 0 and len(cross_tags) == 0):
            return 'pass'

    if 'cross' in tags_str:
        cross_tags = [tag for tag in tags if 'cross' in tag and tag != 'cross interception']
        pass_tags = [tag for tag in tags if 'pass' in tag and tag != 'pass interception']
        assist_tags = [tag for tag in tags if 'assist' in tag]

        if len(cross_tags) > 0 or (len(assist_tags) > 0 and len(pass_tags) == 0):
            return 'cross'

    if 'shot' in tags_str:
        shot_tags = [
            tag for tag in tags
            if 'shot' in tag and tag != 'shot interception' and 'with a shot' not in tag
        ]

        if len(shot_tags) > 0:
            return 'shot'

    if 'dribbl' in tags_str:
        return 'dribble'

    if 'interception' in tags_str:
        return 'interception'

    if 'tackle' in tags_str:
        return 'tackle'

    if 'clearance' in tags_str:
        return 'clearance'

    if 'lost ball' in tags_str or 'bad ball control' in tags_str or 'mistake' in tags_str:
        return 'lost ball'

    if 'recovery' in tags_str:
        return 'recovery'

    if 'rebound' in tags_str:
        return 'rebound'

    if 'foul' in tags_str or 'yc, ' in tags_str or 'rc, ' in tags_str or 'rc for 2 yc' in tags_str or 'yc' == tags_str or 'rc' == tags_str:
        return 'foul'

    if 'challenge' in tags_str:
        return 'challenge'

    if 'own goal' in tags_str:
        return 'own goal'

    if 'save' in tags_str:
        return 'save'

    if 'chance created' in tags_str or 'goal' in tags_str or 'goal-scoring moment' in tags_str:
        goal_tags = [tag for tag in tags if 'goal' == tag or 'goal-scoring moment' in tag or 'chance created' in tag]
        if len(goal_tags) > 0:
            return 'chance'

    if 'opening' in tags_str:
        return 'opening'

    return 'other'


def tagging(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'action_name': 'tags'})
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [x])
    df['action_type'] = df['tags'].apply(tag2type)

    return df


def processing(df: pd.DataFrame) -> pd.DataFrame:
    df = gluing(df)
    df = add_reciever(df)
    df = filter_data(df)
    df = tagging(df)

    return df
