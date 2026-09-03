"""Chronological metadata-only plans, including calendar-year blind reservation."""

import pandas as pd
import pytest

from prooflab.validation.splits import SplitConfig, chronological_split


@pytest.fixture
def timeline():
    return pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")


def config(**updates):
    return SplitConfig(**{"validation_start": "2020-02-01T00:00:00Z",
                          "blind_start": "2020-03-01T00:00:00Z", "max_label_horizon": 3,
                          **updates})


def test_chronological_partition_and_default_calendar_blind(timeline):
    plan = chronological_split(timeline, config(), dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))
    assert plan.train_indices == tuple(range(31))
    assert plan.validation_indices == tuple(range(31, 60))
    assert plan.validation_end == plan.blind_start
    assert "blind_indices" not in plan.model_dump()
    default = chronological_split(timeline, config(blind_start=None),
                                  dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))
    assert default.blind_start == plan.blind_start
    assert default.timeline_checksum == plan.timeline_checksum


@pytest.mark.parametrize("change", ["reversed", "naive", "duplicate", "blind", "empty"])
def test_reject_invalid_research_timeline(timeline, change):
    if change == "reversed":
        timeline = timeline[::-1]
    elif change == "naive":
        timeline = timeline.tz_localize(None)
    elif change == "duplicate":
        timeline = timeline.append(timeline[-1:])
    elif change == "blind":
        timeline = timeline.append(pd.DatetimeIndex(["2020-03-01"], tz="UTC"))
    else:
        timeline = timeline[:0]
    with pytest.raises(ValueError):
        chronological_split(timeline, config(), dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))


def test_random_options_invalid_boundaries_and_leap_year(timeline):
    with pytest.raises(ValueError):
        config(shuffle=True)
    for updates in [{"validation_start": "2019-01-01T00:00:00Z"},
                    {"validation_start": "2020-03-01T00:00:00Z"}]:
        with pytest.raises(ValueError):
            chronological_split(timeline, config(**updates),
                                dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))
    leap = chronological_split(timeline[:31], config(blind_start=None, blind_years=1,
                                validation_start="2020-01-15T00:00:00Z"),
                               dataset_end=pd.Timestamp("2021-02-28", tz="UTC"))
    assert leap.blind_start == pd.Timestamp("2020-02-28", tz="UTC")
