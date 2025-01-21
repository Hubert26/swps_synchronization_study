# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:30:00 2024

@author: Hubert Szewczyk
"""

from datetime import datetime, timedelta
import copy
import logging
from typing import Union

from logger_config import log_message
from classes import Meas
from data_management_functions import validate_meas_data

from utils.signal_utils import interp_signals_uniform_time
from utils.general_utils import group_object_list


# %%
def adjust_timestamps(
    ts1: Union[str, datetime], ts2: Union[str, datetime]
) -> tuple[datetime, datetime]:
    """
    Adjusts two timestamps by setting the older one to zero and adjusting the younger one accordingly.

    Args:
        ts1 (Union[str, datetime]): The first timestamp (ISO format or datetime object).
        ts2 (Union[str, datetime]): The second timestamp (ISO format or datetime object).

    Returns:
        tuple[datetime, datetime]: A tuple containing the adjusted timestamps (older, adjusted_younger).
    """

    def parse_timestamp(ts):
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            return ts
        else:
            raise TypeError(
                "Timestamps must be either strings (ISO format) or datetime objects."
            )

    timestamp1, timestamp2 = parse_timestamp(ts1), parse_timestamp(ts2)

    if timestamp1 < timestamp2:
        older, younger = timestamp1, timestamp2
    else:
        older, younger = timestamp2, timestamp1

    epoch_start = datetime(1970, 1, 1)
    time_difference = (younger - older).total_seconds()

    adjusted_younger = epoch_start + timedelta(seconds=time_difference)

    log_message(
        logging.INFO,
        f"Adjusted timestamps: older={older}, adjusted_younger={adjusted_younger}",
    )

    return epoch_start, adjusted_younger


# %%
def normalize_meas_time(meas_list: list[Meas]) -> list[Meas]:
    """
    Normalizes measurement start times by aligning all starttimes within a group to the earliest found.

    Args:
        meas_list (list[Meas]): List of Meas objects to be normalized.

    Returns:
        list[Meas]: The normalized list of Meas objects with updated starttimes.
    """
    grouped_meas = group_object_list(
        meas_list,
        ["metadata.meas_number", "metadata.condition", "metadata.pair_number"],
    )

    # Iterate over each group of measurements
    for (meas_number, condition, pair_number), group in grouped_meas.items():
        person_meas1 = [meas for meas in group if meas.metadata.gender == "M"]
        person_meas2 = [meas for meas in group if meas.metadata.gender == "F"]

        if not person_meas1 or not person_meas2:
            log_message(
                logging.WARNING,
                "Skipping normalization due to missing gender data",
                meas_number=meas_number,
                condition=condition,
                pair_number=pair_number,
            )
            continue

        all_meas = person_meas1 + person_meas2
        oldest_starttime = min(meas.metadata.starttime for meas in all_meas)

        log_message(
            logging.INFO,
            f"Normalizing measurements for meas_number={meas_number}, condition={condition}, pair_number={pair_number}",
        )

        # Adjust each measurement's timestamp
        for meas in all_meas:
            duration = meas.metadata.endtime - meas.metadata.starttime
            _, meas.metadata.starttime = adjust_timestamps(
                oldest_starttime, meas.metadata.starttime
            )
            meas.metadata.endtime = meas.metadata.starttime + duration

    return meas_list


# %%
def calculate_pair_time_difference(
    meas1: Meas, meas2: Meas, based_on: str = "starttime"
) -> float:
    """
    Calculate the time difference in milliseconds between two signals based on their start or end times.

    Args:
        meas1 (Meas): The first Meas object.
        meas2 (Meas): The second Meas object.
        based_on (str): Specifies whether to calculate the time difference based on 'starttime' or 'endtime'.

    Returns:
        float: The time difference in milliseconds between the two signals.

    Raises:
        ValueError: If based_on is not 'starttime' or 'endtime'.
    """
    if based_on not in ("starttime", "endtime"):
        raise ValueError(
            "The argument 'based_on' must be either 'starttime' or 'endtime'."
        )

    time1 = getattr(meas1.metadata, based_on)
    time2 = getattr(meas2.metadata, based_on)

    time_difference = abs((time2 - time1).total_seconds() * 1000)

    log_message(
        logging.INFO,
        f"Time difference calculated: {time_difference} ms (based on {based_on})",
    )

    return time_difference


# %%
def time_align_pair(meas1: Meas, meas2: Meas) -> None:
    """
    Align the start time of two Meas objects by shifting the younger measurement's x_data.

    Args:
        meas1 (Meas): The first Meas object.
        meas2 (Meas): The second Meas object.

    Raises:
        ValueError: If both Meas objects have the same starttime.
    """
    time_diff_ms = calculate_pair_time_difference(meas1, meas2, based_on="starttime")

    if meas1.metadata.starttime < meas2.metadata.starttime:
        younger_meas, older_meas = meas2, meas1
    elif meas1.metadata.starttime > meas2.metadata.starttime:
        younger_meas, older_meas = meas1, meas2
    else:
        log_message(
            logging.INFO,
            "Both measurements have the same starttime; no alignment needed.",
        )
        return

    log_message(
        logging.INFO,
        f"Aligning measurements: younger ({younger_meas.metadata.starttime}) with older ({older_meas.metadata.starttime}).",
    )

    younger_meas.data.x_data = younger_meas.data.x_data + time_diff_ms
    younger_meas.metadata.starttime = older_meas.metadata.starttime


# %%
def trim_meas(
    meas: Meas, start_ms: float, end_ms: float, starttime: datetime = None
) -> Meas:
    """
    Trims a Meas object to a specified time range.

    Args:
        meas (Meas): The Meas object to be trimmed.
        start_ms (float): Desired start time in milliseconds.
        end_ms (float): Desired end time in milliseconds.
        starttime (datetime, optional): Reference start time.

    Returns:
        Meas: Trimmed Meas object.

    Raises:
        ValueError: If no valid time range is available after trimming.
    """
    meas_copy = copy.deepcopy(meas)
    reference_starttime = starttime or meas.metadata.starttime

    start_offset = (
        reference_starttime - meas.metadata.starttime
    ).total_seconds() * 1000
    adjusted_start_ms, adjusted_end_ms = start_offset + start_ms, start_offset + end_ms

    x_start, x_end = meas_copy.data.x_data[0], meas_copy.data.x_data[-1]
    valid_start, valid_end = (
        max(adjusted_start_ms, x_start),
        min(adjusted_end_ms, x_end),
    )

    if valid_start >= valid_end:
        log_message(
            logging.ERROR,
            f"Invalid trim range: {valid_start} - {valid_end} ms for measurement {meas}",
        )
        raise ValueError(f"No valid data in range {start_ms} to {end_ms} for {meas}")

    meas_copy.trim(valid_start, valid_end)
    log_message(
        logging.INFO, f"Measurement trimmed to range {valid_start} - {valid_end} ms."
    )

    return meas_copy


# %%
def trim_to_common_range(meas_pair: tuple[Meas, Meas]) -> tuple[Meas, Meas]:
    """
    Trims two Meas objects to their common x_data range.

    Args:
        meas_pair (tuple[Meas, Meas]): Two Meas objects to trim.

    Returns:
        tuple[Meas, Meas]: Trimmed Meas objects.

    Raises:
        ValueError: If no common range is found.
    """
    meas1, meas2 = meas_pair
    ref_starttime = min(meas1.metadata.starttime, meas2.metadata.starttime)

    start1 = (
        meas1.metadata.starttime - ref_starttime
    ).total_seconds() * 1000 + meas1.data.x_data[0]
    start2 = (
        meas2.metadata.starttime - ref_starttime
    ).total_seconds() * 1000 + meas2.data.x_data[0]

    end1 = start1 + meas1.data.x_data[-1]
    end2 = start2 + meas2.data.x_data[-1]

    common_start, common_end = max(start1, start2), min(end1, end2)

    if common_start >= common_end:
        raise ValueError("No common range found between the measurements.")

    return trim_meas(meas1, common_start, common_end, ref_starttime), trim_meas(
        meas2, common_start, common_end, ref_starttime
    )


# %%
def interp_meas_pair_uniform_time(
    meas_pair: tuple[Meas, Meas], ix_step: int = 1000
) -> tuple[Meas, Meas]:
    """
    Interpolates two Meas objects to a common uniform time axis.

    Args:
        meas_pair (tuple[Meas, Meas]): Two Meas objects to interpolate.
        ix_step (int): Time step in milliseconds.

    Returns:
        tuple[Meas, Meas]: Interpolated Meas objects.
    """
    meas1, meas2 = trim_to_common_range(meas_pair)

    if not (validate_meas_data(meas1) and validate_meas_data(meas2)):
        log_message(logging.WARNING, "Interpolation validation failed.")
        return None, None

    time_align_pair(meas1, meas2)

    ix, interp_signals = interp_signals_uniform_time(
        [
            (meas1.data.x_data, meas1.data.y_data),
            (meas2.data.x_data, meas2.data.y_data),
        ],
        ix_step=ix_step,
    )

    meas1.data.update(x_data=ix, y_data=interp_signals[0])
    meas2.data.update(x_data=ix, y_data=interp_signals[1])

    log_message(logging.INFO, "Interpolation successful.")

    return meas1, meas2
