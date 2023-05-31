from functools import reduce
from typing import Any, Tuple

import numpy as np


def calculate_possible_starts(*dates: Any, frames_total: int) -> Tuple[Any]:
    """
    Computes the intersection of all input dates and outputs the indices where each index has the next ``frames_total`` timesteps available
    Adapted from: https://github.com/holmdk/IrradianceNet/blob/main/src/data/process_raw_data.py#L56
    Args:
        *dates, Any: datetime arrays or list of datetime.
        frames_total, int: Total number of timesteps to be available.
    Returns:
        possible_indices_dates, Any: Indices for each date in ``dates`` where each index has the next ``frames_total`` timesteps available.
    """
    assert len(dates) >= 1, "You have to provide at least one array!"

    # First compute the dates intersection and get their indices
    date_intersection = reduce(np.intersect1d, dates)
    intersection_indices = [
        np.where(np.isin(date, date_intersection))[0] for date in dates
    ]

    difference_range = np.diff(date_intersection)

    counted = np.zeros(difference_range.shape)
    for idx, time in enumerate(difference_range):
        if idx != counted.shape[0] - 1:
            if time == np.timedelta64(1800000000000, "ns"):
                counted[idx + 1] = 1

    cum_sum = counted.copy()

    for idx, time in enumerate(counted):
        if idx > 0:
            if counted[idx] > 0:
                cum_sum[idx] = cum_sum[idx - 1] + cum_sum[idx]

    possible_indices = np.array(
        np.where(cum_sum >= (frames_total - 1))
    ).ravel()  # 1 since it is index

    # we use the beginning of the sequence as index
    possible_starts = possible_indices - (frames_total - 1)
    possible_starts = possible_starts.astype("int")

    possible_starts.sort()

    # Return possible indices from the original dates
    possible_indices_dates = tuple(
        [
            intersection_date[possible_starts]
            for intersection_date in intersection_indices
        ]
    )
    if len(possible_indices_dates) == 1:
        return possible_indices_dates[0]
    return possible_indices_dates
