"""Signal preprocessing for AQUINAS sensor records."""

from aquinas_toolkit.preprocessing.alignment import AlignedEvent, align_event_group, synchro_indices
from aquinas_toolkit.preprocessing.core import (
    LoadedEventGroup,
    find_events,
    load_event_group,
    load_timestamp_query_frames,
)
from aquinas_toolkit.preprocessing.organizer import OrganizerQueryResult, run_organizer_query
from aquinas_toolkit.preprocessing.pipeline import export_aligned_event, run_preprocessing
from aquinas_toolkit.preprocessing.zeroing import (
    zero_loaded_event_group,
    zero_waveform,
)
from aquinas_toolkit.preprocessing.signals import (
    SIGNAL_FILTER_METHODS,
    bandpass_filter_waveform_matrix,
    filter_loaded_event_group,
    find_common_sensor_events,
    load_common_event_waveform_matrix,
)
from aquinas_toolkit.preprocessing.duration import (
    filter_records_by_min_duration,
    summarize_min_duration_filter,
)


__all__ = [
    "AlignedEvent",
    "LoadedEventGroup",
    "OrganizerQueryResult",
    "align_event_group",
    "export_aligned_event",
    "find_events",
    "load_event_group",
    "load_timestamp_query_frames",
    "run_preprocessing",
    "run_organizer_query",
    "synchro_indices",
    "zero_loaded_event_group",
    "zero_waveform",
    "SIGNAL_FILTER_METHODS",
    "bandpass_filter_waveform_matrix",
    "filter_loaded_event_group",
    "filter_records_by_min_duration",
    "find_common_sensor_events",
    "load_common_event_waveform_matrix",
    "summarize_min_duration_filter",
]
