from abc import ABC, abstractmethod

import pandas as pd
import typing


def assert_subset(required_columns: typing.List[str], all_columns: typing.List[str]):
    """
    Asserts that one list of strings is a subset of the other list.

    Args:
        required_columns (List[str]): subset
        all_columns (List[str]): full set
    """
    assert set(required_columns).issubset(set(all_columns)
                                          ), 'Required columns are not in the dataframe.'


class FeatureGenerator(ABC):
    def __init__(self, name: str, required_columns: typing.List[str]):
        self.name = name
        self.required_columns = required_columns

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def schema(self):
        pass


class HighTip(FeatureGenerator):
    def __init__(self):
        super(HighTip, self).__init__(
            'high_tip', ['tip_amount', 'fare_amount'])

    def compute(self, df: pd.DataFrame, tip_fraction: float = 0.2) -> pd.DataFrame:
        assert_subset(self.required_columns, df.columns)
        tip_fraction_col = df.tip_amount / df.fare_amount
        feature_df = pd.DataFrame(
            {'tip_fraction': tip_fraction_col > tip_fraction})
        return feature_df[self.schema().keys()]

    def schema(self) -> dict:
        return {'tip_fraction': bool}


class Pickup(FeatureGenerator):
    def __init__(self):
        super(Pickup, self).__init__('pickup', ['tpep_pickup_datetime'])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_subset(self.required_columns, df.columns)
        pickup_weekday = df.tpep_pickup_datetime.dt.weekday
        pickup_hour = df.tpep_pickup_datetime.dt.hour
        pickup_minute = df.tpep_pickup_datetime.dt.minute
        work_hours = (pickup_weekday >= 0) & (pickup_weekday <= 4) & (
            pickup_hour >= 8) & (pickup_hour <= 18)
        feature_df = pd.DataFrame({'pickup_weekday': pickup_weekday,
                                   'pickup_hour': pickup_hour, 'pickup_minute': pickup_minute, 'work_hours': work_hours})
        return feature_df[self.schema().keys()]

    def schema(self) -> dict:
        return {'pickup_weekday': int, 'pickup_hour': int, 'pickup_minute': int, 'work_hours': bool}


class Trip(FeatureGenerator):
    def __init__(self):
        super(Trip, self).__init__('trip', ['tpep_dropoff_datetime',
                                            'tpep_pickup_datetime', 'trip_distance', 'passenger_count'])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_subset(self.required_columns, df.columns)
        trip_time = (df.tpep_dropoff_datetime -
                     df.tpep_pickup_datetime).dt.seconds
        trip_speed = df.trip_distance / (trip_time + 1e7)
        feature_df = pd.DataFrame({'trip_time': trip_time, 'trip_speed': trip_speed,
                                   'trip_distance': df.trip_distance, 'passenger_count': df.passenger_count})
        return feature_df[self.schema().keys()]

    def schema(self) -> dict:
        return {'passenger_count': int, 'trip_distance': float, 'trip_time': int, 'trip_speed': float}


class Categorical(FeatureGenerator):
    def __init__(self):
        super(Categorical, self).__init__('categorical', ['PULocationID',
                                                          'DOLocationID', 'RatecodeID'])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_subset(self.required_columns, df.columns)
        return df[self.schema().keys()]

    def schema(self) -> dict:
        return {col: int for col in self.required_columns}
