import h5py
from typing import Optional, Union
import numpy as np
from ..._scipp import core as sc
from datetime import datetime


class BadSource(Exception):
    pass


def _get_units(dataset: h5py.Dataset) -> Optional[str]:
    try:
        units = dataset.attrs["units"]
    except AttributeError:
        return None
    return _ensure_str(units)


def _ensure_str(str_or_bytes: Union[str, bytes]) -> str:
    try:
        str_or_bytes = str(str_or_bytes, encoding="utf8")  # type: ignore
    except TypeError:
        pass
    return str_or_bytes


def _get_pulse_time_offset(pulse_time_dataset: h5py.Dataset) -> Optional[str]:
    try:
        pulse_offset_iso8601 = pulse_time_dataset.attrs["offset"]
    except KeyError:
        return None
    return _ensure_str(pulse_offset_iso8601)


def _check_for_missing_fields(group: h5py.Group) -> Optional[str]:
    error_message = None
    required_fields = (
        "event_time_zero",
        "event_index",
        "event_id",
        "event_time_offset",
    )
    for field in required_fields:
        if field not in group:
            error_message += f"Unable to load data from NXevent_data " \
                             f"at '{group.name}' due to missing '{field}'" \
                             f" field\n"
    return error_message


def _iso8601_to_datetime(iso8601: str) -> Optional[datetime]:
    try:
        return datetime.strptime(
            iso8601.translate(str.maketrans('', '', ':-Z')),
            "%Y%m%dT%H%M%S.%f")
    except ValueError:
        # Did not understand the format of the input string
        return None


def load_event_group(group: h5py.Group) -> sc.Variable:
    error_msg = _check_for_missing_fields(group)
    if error_msg is not None:
        raise BadSource(error_msg)

    # There is some variation in the last recorded event_index in files
    # from different institutions. We try to make sure here that it is what
    # would be the first index of the next pulse.
    # In other words, ensure that event_index includes the bin edge for
    # the last pulse.
    event_index = group["event_index"][...].astype(np.int64)
    if event_index[-1] < group["event_id"].len():
        event_index = np.append(
            event_index,
            np.array([group["event_id"].len() - 1]).astype(event_index.dtype),
        )
    else:
        event_index[-1] = group["event_id"].len()

    number_of_events = event_index[-1]
    event_time_offset = sc.Variable(
        ['event'],
        values=group["event_time_offset"][...],
        dtype=group["event_time_offset"].dtype.type)
    event_id = sc.Variable(
        ['event'], values=group["event_id"][...],
        dtype=np.int32)  # assume int32 is safe for detector ids
    event_time_zero = sc.Variable(['pulse'],
                                  values=group["event_time_zero"][...],
                                  dtype=group["event_time_zero"].dtype.type)
    pulse_time_offset = _get_pulse_time_offset(group["event_time_zero"])

    unix_epoch = datetime(1970, 1, 1)
    if pulse_time_offset is not None and pulse_time_offset != unix_epoch:
        # TODO correct for time offset, convert to relative to unix epoch
        #   or do we want to cast pulse times to datetime objects anyway?
        #   (is microsecond precision sufficient?)
        NotImplementedError(
            "Found offset for pulse times but dealing with this "
            "is not implemented yet")

    # The end index for a pulse is the start index of the next pulse
    begin_indices = sc.Variable(['pulse'], values=event_index[:-1])
    end_indices = sc.Variable(['pulse'], values=event_index[1:])

    # Weights are not stored in NeXus, so use 1s
    weights = sc.Variable(['event'],
                          values=np.ones(event_id.shape),
                          dtype=np.float32)
    data = sc.DataArray(data=weights,
                        coords={
                            'tof': event_time_offset,
                            'detector-id': event_id
                        })
    try:
        events = sc.DataArray(data=sc.bins(begin=begin_indices,
                                           end=end_indices,
                                           dim='event',
                                           data=data),
                              coords={'pulse-time': event_time_zero})
    except IndexError:
        # For example found max uint64 at end of some event_index
        # datasets in SNS files
        raise BadSource("Unexpected values for event indices in "
                        "event_index dataset")

    if "detector_numbers" in group:
        detector_numbers = sc.Variable(
            dims=['detector-id'],
            values=group['detector_numbers'][...].flatten(),
            dtype=np.int32)
    else:
        # No detector_numbers dataset so we'll have to find what range of
        # detectors numbers are in the dataset
        detector_numbers = sc.Variable(dims=['detector-id'],
                                       values=np.arange(event_index.min(),
                                                        event_index.max() + 1,
                                                        dtype=np.int32))
    # Events in the NeXus file are effectively binned by pulse
    # (because they are recorded chronologically)
    # but for reduction it is more useful to bin by detector id
    events = sc.bin(events, groups=[detector_numbers], erase=['pulse'])

    print(f"Loaded event data from {group.name} containing "
          f"{number_of_events} events")

    return events
