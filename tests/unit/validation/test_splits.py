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
    assert plan.train_indices == tuple(range(28))
    assert plan.validation_indices == tuple(range(31, 57))
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


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_walk_forward_complete_windows(timeline, mode):
    from prooflab.validation.splits import WalkForwardConfig, walk_forward
    cfg = WalkForwardConfig(**config().model_dump(), mode=mode, training_bars=20,
                            validation_bars=10, step_bars=10)
    plans = walk_forward(timeline, cfg, dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))
    assert len(plans) == 2
    assert plans[0].train_indices == tuple(range(11, 28))
    assert plans[1].train_indices == tuple(range(21 if mode == "rolling" else 11, 38))
    assert plans[0].validation_indices == tuple(range(31, 38))
    assert plans[1].validation_indices == tuple(range(41, 48))
    assert plans[0].validation_end <= plans[1].validation_start


def test_invalid_walk_forward_windows(timeline):
    from prooflab.validation.splits import WalkForwardConfig, walk_forward
    for kwargs in [{"mode": "rolling"}, {"step_bars": 1}]:
        with pytest.raises(ValueError):
            WalkForwardConfig(**config().model_dump(), validation_bars=10, **kwargs)
    for kwargs in [{"training_bars": 40}, {"validation_bars": 50}]:
        cfg = WalkForwardConfig(**{**config().model_dump(), "validation_bars": 10, **kwargs})
        with pytest.raises(ValueError):
            walk_forward(timeline, cfg, dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))



def test_purge_full_horizons_and_irregular_times(timeline):
    index = timeline.delete([4, 5, 12])
    horizons = pd.Series(1, index=index)
    horizons.iloc[25] = 3
    plan = chronological_split(index, config(), dataset_end=pd.Timestamp("2022-03-01", tz="UTC"),
                               horizon_bars=horizons)
    assert 25 in plan.purged_training
    assert 26 in plan.embargoed_training
    assert 27 in plan.purged_training
    assert all(index[i + horizons.iloc[i]] < plan.validation_start for i in plan.train_indices)
    assert all(index[i + horizons.iloc[i]] < plan.blind_start for i in plan.validation_indices)
    assert plan.purged_validation == (len(index) - 1,)


@pytest.mark.parametrize("change", ["unaligned", "zero", "large", "float"])
def test_invalid_full_horizons(timeline, change):
    horizons = pd.Series(3, index=timeline)
    if change == "unaligned":
        horizons = horizons.iloc[::-1]
    elif change == "zero":
        horizons.iloc[0] = 0
    elif change == "large":
        horizons.iloc[0] = 4
    else:
        horizons = horizons.astype(float)
    with pytest.raises(ValueError, match="Horizon bars"):
        chronological_split(timeline, config(), dataset_end=pd.Timestamp("2022-03-01", tz="UTC"),
                            horizon_bars=horizons)



def test_both_embargo_buffers_and_exact_intervals(timeline):
    from prooflab.validation.splits import WalkForwardConfig, walk_forward
    cfg = WalkForwardConfig(**config(embargo_bars=5).model_dump(), validation_bars=8, step_bars=8)
    plans = walk_forward(timeline, cfg, dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))
    assert plans[0].train_indices == tuple(range(26))
    third = plans[2]
    post = [span for span in third.embargo_intervals if span.reason == "after_validation"]
    assert any(span.start_position == 39 and span.stop_position == 44 for span in post)
    assert not set(third.train_indices).intersection(range(39, 44))
    assert all(span.configured_bars == 5 for span in third.embargo_intervals)
    for span in third.embargo_intervals:
        assert span.start == timeline[span.start_position]
        assert span.end == (timeline[span.stop_position] if span.stop_position < len(timeline)
                            else third.blind_start)
    static = chronological_split(timeline, config(embargo_bars=5),
                                 dataset_end=pd.Timestamp("2022-03-01", tz="UTC"))
    assert static.validation_indices[-1] == 54
    assert static.embargoed_validation == (55, 56)
    assert static.embargo_bars >= static.configuration["max_label_horizon"]


def test_embargo_cannot_be_disabled_or_shortened():
    for length in [0, 1, 2]:
        with pytest.raises(ValueError):
            config(embargo_bars=length)
