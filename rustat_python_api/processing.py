import pandas as pd
import numpy as np

from .config import columns, id2type


def process_list(x: pd.Series):
    lst = x.dropna().unique().tolist()
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
    df['receiver_id'] = df['player_id'].shift(-1)
    df['receiver_name'] = df['player_name'].shift(-1)

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
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

    return df[(~df['possession_number'].isna()) | (df['second'] != 0)][columns].reset_index(drop=True)


def tagging(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'action_name': 'sub_tags', 'action_id': 'sub_tags_ids'})
    df['sub_tags'] = df['sub_tags'].apply(lambda x: x if isinstance(x, list) else [x])
    df['sub_tags_ids'] = df['sub_tags_ids'].apply(
        lambda x:
        list(set([int(t) // 1000 for t in x]))
        if isinstance(x, list)
        else [int(x) // 1000]
    )
    df['sub_tags_ids'] = df['sub_tags_ids'].apply(lambda x: [id2type[t] for t in x])
    df = df.rename(columns={'sub_tags_ids': 'tags'})

    return df


def processing(df: pd.DataFrame) -> pd.DataFrame:
    df = gluing(df)
    df = add_reciever(df)
    df = filter_data(df)
    df = tagging(df)

    return df
