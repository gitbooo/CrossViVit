import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature(ABC):
    @abstractmethod
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency(frequency: str) -> List[TimeFeature]:
    """
    Frequency string: [number][granularity], such as '12H', '5min', '1D', etc.
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }
    offset = to_offset(frequency)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]
    raise RuntimeError(
        f"""
    Unsupported frequency {frequency}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    )


def time_features(dates, time_encoding=True, frequency: str = "t"):
    if not time_encoding:
        dates["month"] = dates.date.apply(lambda row: row.month, 1)
        dates["day"] = dates.date.apply(lambda row: row.day, 1)
        dates["weekday"] = dates.date.apply(lambda row: row.weekday(), 1)
        dates["hour"] = dates.date.apply(lambda row: row.hour, 1)
        if frequency == "t":
            dates["minute"] = dates.date.apply(lambda row: row.minute, 1)
            dates["minute"] = dates.minute.map(lambda x: x // 15)
        FREQUENCY_DICT = {
            "y": [],
            "m": ["month"],
            "w": ["month"],
            "d": ["month", "day", "weekday"],
            "b": ["month", "day", "weekday"],
            "h": ["month", "day", "weekday", "hour"],
            "t": ["month", "day", "weekday", "hour", "minute"],
        }
        return dates[FREQUENCY_DICT[frequency.lower()]].values
    dates = pd.to_datetime(dates.date.values)
    return np.vstack([feat(dates) for feat in time_features_from_frequency(frequency)]).transpose(1, 0)
