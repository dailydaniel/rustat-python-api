import pandas as pd
import numpy as np
from ast import literal_eval

from .pc_adder import PitchControlAdder
from .tr_adder import TrackingFeaturesAdder


class MatchInferLoader:
    def __init__(
            self,
            events: pd.DataFrame, tracking: pd.DataFrame, ball: pd.DataFrame,
            modes: list[str], rads: list[int],
            radii: list[int], cone_degrees: list[int], k_list: list[int],
    ):
        self.events = events
        self.tracking = tracking
        self.ball = ball
        self.modes = modes
        self.rads = rads
        self.radii = radii
        self.cone_degrees = cone_degrees
        self.k_list = k_list

    def _save_index(self):
        self.events['orig_index'] = self.events.index

    def _process_events(self):
        self.events = process_events_after_loading(self.events)

    def _add_pc_features(self):
        pc_adder = PitchControlAdder(self.events, self.tracking, self.ball, device="mps", backend="pt")
        pc_adder.run(modes=self.modes, rads=self.rads)
        self.events = pc_adder.events

    def _add_tr_features(self):
        tr_adder = TrackingFeaturesAdder(self.events, self.tracking, self.ball)
        tr_adder.run(
            radii=self.radii,
            cone_degrees=self.cone_degrees,
            k_list=self.k_list
        )
        self.events = tr_adder.events

    def fit(self, inplace: bool = False):
        self._process_events()
        self._add_pc_features()
        self._add_tr_features()

        if not inplace:
            return self.events

    def get_tracking_columns(self, events: pd.DataFrame = None) -> list[str]:
        if events is None:
            events = self.events

        pc_columns = [column for column in events.columns if column.startswith("pc_")]
        tr_columns = [column for column in events.columns if column.startswith("tf_")]

        return pc_columns + tr_columns


def process_events_after_loading(events: pd.DataFrame) -> pd.DataFrame:
    events = events[events['half'].isin([1, 2])]

    events[[
        'pos_dest_x', 'pos_dest_y'
    ]] = events[[
        'pos_dest_x', 'pos_dest_y'
    ]].apply(pd.to_numeric, errors='coerce')

    events['sub_tags'] = events['sub_tags'].astype(str).apply(literal_eval)
    events['tags'] = events['tags'].astype(str).apply(literal_eval)
    events['possession_team_name'] = events['possession_team_name'].astype(str).apply(
        lambda x: literal_eval(x) if isinstance(x, str) and '[' in x else x
    )

    events['team_id'] = events['team_id'].astype(np.int64)

    return events
