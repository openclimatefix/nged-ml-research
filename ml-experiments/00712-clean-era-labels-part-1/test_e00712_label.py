"""Tests of the stage 1 labelling script of experiment 712: no test makes a call.

The parser test over the stored replies of the cost trial counts valid and invalid replies and
prints schema errors only. No test prints, displays, or summarises a label, a reason, or a reply.
"""

# ruff: noqa: D103, ANN001, ANN202 - a test's name documents it, as for `tests/` (pyproject.toml)
import json
import shutil
import sys
import types
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).resolve().parent
for folder in (HERE, *HERE.parent.glob("00709-*")):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

features = pytest.importorskip("e00709_features")
asof = pytest.importorskip("e00712_asof")
label = pytest.importorskip("e00712_label")
prompts = pytest.importorskip("e00712_prompt")

N_DAYS = 120
SITE = 0


def _candidate(**changes: object) -> prompts.Candidate:
    """A candidate of the first Primary: onset 28, alarm 30, closed by a return on day 45."""
    base = prompts.Candidate(
        site=SITE, episode_id="stat0", detector="stat", onset=28, alarm=30, end=45,
        ended_by="return", stat_alarm=None,
    )  # fmt: skip
    return replace(base, **changes)


def _bank(*, read_day: int = 57) -> asof.DetectorBank:
    """A bank with a delta that grows by 0.1 a day and one readable 12-statistic vector."""
    ramp = np.tile((np.arange(N_DAYS) * 0.1)[:, None], (1, features.N_FEATURES))
    return asof.DetectorBank(
        n_days=N_DAYS,
        spread=np.ones((2, features.N_FEATURES)),
        delta={(SITE, "stat"): ramp},
        vectors={(SITE, "stat", 30): (read_day, np.full(12, 5.0))},
    )


def _resolved(kind: str, day: int, name: str | None, confidence: float) -> asof.Resolved:
    return asof.Resolved(kind=kind, day=day, label=name, confidence=confidence)


def _table(
    answers: list[asof.Resolved], *, candidate: prompts.Candidate | None = None, **kw: object
):
    c = candidate or _candidate()
    return asof.build_table(
        arm="test", candidates=[c], answers={c.key: answers}, bank=_bank(), n_sites=2, **kw
    )


# ---------------------------------------------------------------- the schema


def _reply(**changes: object) -> str:
    body: dict[str, object] = {
        "label": "era", "onset_offset_days": -3, "confidence": 0.9, "neighbour_mirror": "no",
        "shape": "solar", "register_row": None, "reason": "a few words",
    }  # fmt: skip
    return json.dumps(body | changes)


def test_parse_reply_accepts_a_valid_reply_and_a_code_fence() -> None:
    assert label.parse_reply(text=_reply()).label == "era"
    assert label.parse_reply(text="```json\n" + _reply() + "\n```").confidence == 0.9


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"label": "maybe"}, "label"),
        ({"onset_offset_days": 2}, "onset_offset_days"),
        ({"onset_offset_days": -1.5}, "onset_offset_days"),
        ({"onset_offset_days": True}, "onset_offset_days"),
        ({"confidence": 1.2}, "confidence"),
        ({"confidence": "high"}, "confidence"),
        ({"neighbour_mirror": "maybe"}, "neighbour_mirror"),
        ({"shape": "round"}, "shape"),
        ({"register_row": 1}, "register_row"),
        ({"reason": 7}, "reason"),
    ],
)
def test_parse_reply_names_the_bad_field_and_never_a_value(change: dict, field: str) -> None:
    with pytest.raises(label.SchemaError) as error:
        label.parse_reply(text=_reply(**change))
    assert field in str(error.value)
    for value in change.values():
        if isinstance(value, str) and len(value) > 3:
            assert value not in str(error.value)


def test_parse_reply_register_row_must_be_a_shown_row() -> None:
    assert label.parse_reply(text=_reply(register_row=2), n_register_rows=2).register_row == 2
    with pytest.raises(label.SchemaError):
        label.parse_reply(text=_reply(register_row=3), n_register_rows=2)


def test_parse_reply_rejects_missing_and_extra_keys_and_non_json() -> None:
    body = json.loads(_reply())
    del body["shape"]
    for text in (json.dumps(body), json.dumps({**json.loads(_reply()), "x": 1}), "not json", "[1]"):
        with pytest.raises(label.SchemaError):
            label.parse_reply(text=text)


def test_register_row_may_be_absent_and_a_long_reason_is_cut_and_flagged() -> None:
    body = json.loads(_reply())
    del body["register_row"]
    answer = label.parse_reply(text=json.dumps(body))
    assert answer.register_row is None
    assert answer.parser_version == label.PARSER_LENIENT
    assert not answer.reason_truncated
    with pytest.raises(label.SchemaError, match="keys differ"):
        label.parse_reply(text=json.dumps(body), strict=True)
    long = label.parse_reply(text=_reply(reason="word " * 61))
    assert long.reason_truncated
    assert len(long.reason.split()) == label.MAX_REASON_WORDS
    assert long.as_json()["parser_version"] == label.PARSER_LENIENT
    with pytest.raises(label.SchemaError, match="reason"):
        label.parse_reply(text=_reply(reason="word " * 61), strict=True)
    with pytest.raises(label.SchemaError, match="confidence"):
        label.parse_reply(text=_reply(reason="word " * 61, confidence=1.5))
    assert label.parse_reply(text=_reply(), strict=True).parser_version == label.PARSER_STRICT


def test_reparsing_a_log_leaves_the_original_entries_unchanged() -> None:
    good = {"tag": "label", "status": "ok", "call_id": "llm:a_alarm", "answer": None,
            "raw_output": _reply(reason="word " * 61)}  # fmt: skip
    retry = {**good, "raw_output": _reply(label="switch"), "attempt": 1}
    broken = {**good, "call_id": "llm:b_alarm", "raw_output": "not json"}
    probe = {
        "tag": "billing-probe",
        "status": "ok",
        "call_id": "llm:billing_probe_0",
        "answer": None,
    }
    entries = [good, retry, broken, probe]
    before = json.dumps(entries)
    out = label.reparse_log(entries=entries)
    assert json.dumps(entries) == before
    assert out[0]["answer"]["reason_truncated"]
    assert out[0]["parser_version"] == label.PARSER_LENIENT
    assert out[0]["original_answer"] is None
    assert out[1]["answer"] is None  # a live run would not have asked again
    assert out[2]["answer"] is None
    assert out[2]["parse_error"].startswith("not valid JSON")
    assert out[3]["answer"] is None


def test_stored_trial_replies_count_valid_and_invalid_only() -> None:
    if not label.TRIAL_LOG_PATH.exists():
        pytest.skip("the stored trial log is not on this machine")
    valid, invalid, errors = label.schema_report(log_path=label.TRIAL_LOG_PATH, strict=True)
    print(f"stored trial replies: {valid} satisfy the schema, {invalid} fail it")
    for message, count in errors.items():
        print(f"  {count} x {message}")
    assert valid + invalid == 21


# ---------------------------------------------------------------- the confidence rule and states


@pytest.mark.parametrize(
    ("name", "confidence", "state"),
    [
        ("era", 0.8, asof.STATE_ERA),
        ("era", 0.79, asof.STATE_UNKNOWN),
        ("switch", 0.95, asof.STATE_SWITCH),
        ("noise", 0.8, asof.STATE_NONE),
        ("noise", 0.5, asof.STATE_UNKNOWN),
        ("undecided", 1.0, asof.STATE_UNKNOWN),
        (None, 0.0, asof.STATE_UNKNOWN),
    ],
)
def test_confidence_rule(name: str | None, confidence: float, state: int) -> None:
    assert asof.class_of(label=name, confidence=confidence) == state


def test_timeline_is_absent_before_the_alarm_and_unknown_until_an_answer() -> None:
    line = asof.class_timeline(
        candidate=_candidate(), resolved=[_resolved("plus7", 37, "era", 0.9)], n_days=N_DAYS
    )
    assert (line[:30] == asof.ABSENT).all()
    assert (line[30:37] == asof.STATE_UNKNOWN).all()
    assert (line[37:] == asof.STATE_ERA).all()


def test_an_era_survives_a_weak_answer_only_when_sticky() -> None:
    answers = [_resolved("alarm", 30, "era", 0.9), _resolved("plus7", 37, "undecided", 0.9)]
    sticky = asof.class_timeline(candidate=_candidate(), resolved=answers, n_days=N_DAYS)
    literal = asof.class_timeline(
        candidate=_candidate(), resolved=answers, n_days=N_DAYS, sticky_era=False
    )
    assert sticky[40] == asof.STATE_ERA
    assert literal[40] == asof.STATE_UNKNOWN
    overturn = [*answers[:1], _resolved("plus7", 37, "switch", 0.85)]
    assert (
        asof.class_timeline(candidate=_candidate(), resolved=overturn, n_days=N_DAYS)[40]
        == asof.STATE_SWITCH
    )


# ---------------------------------------------------------------- the schedule


def test_closure_day_follows_the_way_the_episode_closed() -> None:
    assert _candidate().closure_day == 45
    assert _candidate(ended_by="superseded").closure_day == 46
    assert _candidate(end=-1, ended_by="open").closure_day is None


def test_scheduled_checks_drop_after_the_closure_and_keep_the_alarm_check() -> None:
    kinds = [
        c.kind
        for c in asof.scheduled_checks(candidate=_candidate(), offsets=(0, 7, 21), last_day=100)
    ]
    assert kinds == ["alarm", "plus7"]
    early = _candidate(end=31)
    kinds = [
        c.kind for c in asof.scheduled_checks(candidate=early, offsets=(0, 7, 21), last_day=100)
    ]
    assert kinds == ["alarm"]


def test_next_check_walks_the_schedule_and_stops_early_on_two_agreeing_answers() -> None:
    c = _candidate(end=-1, ended_by="open")
    done: list[asof.Resolved] = []
    seen = []
    while (check := asof.next_check(candidate=c, resolved=done, last_day=100)) is not None:
        seen.append(check.kind)
        done.append(_resolved(check.kind, check.day, "era", 0.9))
    assert seen == ["alarm", "plus7"]
    weak = []
    seen = []
    while (check := asof.next_check(candidate=c, resolved=weak, last_day=100)) is not None:
        seen.append(check.kind)
        weak.append(_resolved(check.kind, check.day, "era", 0.6))
    assert seen == ["alarm", "plus7", "plus21"]


def test_next_check_runs_the_closure_check_while_era_or_unknown_and_not_otherwise() -> None:
    c = _candidate()
    agreed = [_resolved("alarm", 30, "era", 0.9), _resolved("plus7", 37, "era", 0.9)]
    assert asof.next_check(candidate=c, resolved=agreed, last_day=100).kind == "closure"
    assert asof.next_check(candidate=c, resolved=agreed, last_day=100).day == 45
    switched = [_resolved("alarm", 30, "switch", 0.9), _resolved("plus7", 37, "switch", 0.9)]
    assert asof.next_check(candidate=c, resolved=switched, last_day=100) is None
    noise = [_resolved("alarm", 30, "noise", 0.9), _resolved("plus7", 37, "noise", 0.9)]
    assert asof.next_check(candidate=c, resolved=noise, last_day=100) is None
    finished = [*agreed, _resolved("closure", 45, "era", 0.9)]
    assert asof.next_check(candidate=c, resolved=finished, last_day=100) is None
    assert asof.next_check(candidate=c, resolved=agreed, last_day=44) is None


def test_the_schedule_never_depends_on_a_fact_revealed_later() -> None:
    for candidate in (_candidate(), _candidate(ended_by="superseded"), _candidate(end=33)):
        for day in range(30, 90):
            seen = asof.visible_candidate(candidate=candidate, day=day)
            full = [
                (c.kind, c.day)
                for c in asof.all_possible_checks(
                    candidate=candidate, offsets=(0, 7, 21), last_day=200
                )
                if c.day <= day
            ]
            then = [
                (c.kind, c.day)
                for c in asof.all_possible_checks(candidate=seen, offsets=(0, 7, 21), last_day=200)
                if c.day <= day
            ]
            assert full == then


# ---------------------------------------------------------------- the as-of table


def test_det_table_is_era_from_the_alarm_day_and_never_ends() -> None:
    c = _candidate()
    table = asof.build_table(
        arm="det", candidates=[c], answers=asof.det_answers(candidates=[c]), bank=_bank(), n_sites=2
    )
    state, cols = table.columns["state"][SITE], table.columns
    assert (state[:30] == asof.STATE_NONE).all()
    assert (state[30:] == asof.STATE_ERA).all()
    assert (cols["state"][1] == 0).all()
    weights = features.BAND_WEIGHTS * np.ones(features.N_FEATURES)
    live = np.full(features.N_FEATURES, 0.1 * 35)
    assert cols["anchor"][SITE, 35] == pytest.approx(weights @ live)
    frozen = np.full(features.N_FEATURES, 0.1 * min(30 + asof.SETTLING_DAYS, 45))
    assert cols["anchor"][SITE, 100] == pytest.approx(weights @ frozen)
    assert cols["era_id"][SITE, 56] == 0
    assert cols["era_id"][SITE, 57] == 1
    assert cols["onset_day"][SITE, 35] == 28
    assert cols["confirm_day"][SITE, 35] == 30
    row = table.episodes.row(0, named=True)
    assert (row["onset"], row["alarm"], row["end"], row["first_era"]) == (28, 30, 45, 30)


def test_unknown_then_switch_then_none_after_the_end() -> None:
    answers = [_resolved("alarm", 30, "era", 0.5), _resolved("plus7", 37, "switch", 0.9)]
    table = _table(answers)
    state = table.columns["state"][SITE]
    assert (state[30:37] == asof.STATE_UNKNOWN).all()
    assert (state[37:46] == asof.STATE_SWITCH).all()
    assert (state[46:] == asof.STATE_NONE).all()
    assert table.columns["depth"][SITE, 40] != 0.0
    assert table.columns["anchor"][SITE, 40] == 0.0
    assert table.columns["withdrawn"][SITE].sum() == 0
    row = table.episodes.row(0, named=True)
    assert (row["end"], row["first_era"]) == (45, -1)


def test_an_overturned_era_sets_withdrawn_and_widens_the_episode_end() -> None:
    c = _candidate(ended_by="superseded", end=45)
    answers = [_resolved("alarm", 30, "era", 0.9), _resolved("closure", 46, "switch", 0.9)]
    table = _table(answers, candidate=c)
    withdrawn = table.columns["withdrawn"][SITE]
    assert withdrawn[46] == 1
    assert withdrawn.sum() == 1
    state = table.columns["state"][SITE]
    assert (state[30:46] == asof.STATE_ERA).all()
    assert state[46] == asof.STATE_NONE
    assert table.columns["anchor"][SITE, 47] == 0.0
    row = table.episodes.row(0, named=True)
    assert row["end"] == 46
    assert row["first_era"] == 30


def test_noise_keeps_its_episode_row_with_state_none() -> None:
    table = _table([_resolved("alarm", 30, "noise", 0.9)])
    assert (table.columns["state"][SITE] == asof.STATE_NONE).all()
    assert (table.columns["onset_day"][SITE] == -1).all()
    row = table.episodes.row(0, named=True)
    assert (row["end"], row["first_era"]) == (45, -1)


def test_a_candidate_with_no_valid_reply_stays_unknown_and_open() -> None:
    table = _table([])
    state = table.columns["state"][SITE]
    assert (state[30:46] == asof.STATE_UNKNOWN).all()
    assert (table.episodes["end"] == 45).all()


def test_a_label_reaches_only_days_on_or_after_its_check_day() -> None:
    answers = [_resolved("alarm", 30, "era", 0.9), _resolved("plus7", 37, "switch", 0.9)]
    full = _table(answers)
    for cut in (31, 36, 37, 50):
        earlier = _table([r for r in answers if r.day < cut])
        for name in asof.COLUMN_NAMES:
            if name == "era_id":
                continue
            assert np.array_equal(full.columns[name][:, :cut], earlier.columns[name][:, :cut]), name


def test_a_second_era_takes_a_later_id_and_a_withdrawal_returns_to_the_earlier_one() -> None:
    first = _candidate(episode_id="stat0", onset=28, alarm=30, end=34, ended_by="return")
    second = _candidate(episode_id="stat1", onset=60, alarm=62, end=-1, ended_by="open")
    ramp = np.tile((np.arange(N_DAYS) * 0.1)[:, None], (1, features.N_FEATURES))
    bank = asof.DetectorBank(
        n_days=N_DAYS,
        spread=np.ones((2, features.N_FEATURES)),
        delta={(SITE, "stat"): ramp},
        vectors={
            (SITE, "stat", 30): (31, np.full(12, 5.0)),
            (SITE, "stat", 62): (63, np.full(12, 9.0)),
        },
    )
    answers = {
        first.key: [_resolved("alarm", 30, "era", 0.9)],
        second.key: [_resolved("alarm", 62, "era", 0.9), _resolved("plus7", 69, "switch", 0.9)],
    }
    table = asof.build_table(
        arm="test", candidates=[first, second], answers=answers, bank=bank, n_sites=2
    )
    ids = table.columns["era_id"][SITE]
    assert ids[40] == 1
    assert ids[65] == 2
    assert ids[70] == 1
    assert table.columns["withdrawn"][SITE, 69] == 1


def test_ceiling_and_truth_use_the_era_table_and_the_verified_events() -> None:
    era = pl.DataFrame(
        {"primary": [features.PRIMARIES[SITE]], "era_index": [1], "register_date": [None],
         "telemetry_date": [features.day_of(index=29)]},
        schema_overrides={"register_date": pl.Date},
    )  # fmt: skip
    events = pl.DataFrame(
        schema={"source": pl.String, "site": pl.String, "onset_utc": pl.Datetime("us", "UTC"),
                "end_utc": pl.Datetime("us", "UTC")}
    )  # fmt: skip
    a, b = _candidate(), _candidate(episode_id="stat9", onset=90, alarm=92)
    truth = asof.true_classes(candidates=[a, b], era_table=era, known_events=events)
    assert truth == {a.key: "era", b.key: "noise"}
    answers = asof.ceiling_answers(candidates=[a, b], truth=truth)
    assert answers[a.key][0].label == "era"
    table = asof.build_table(
        arm="ceiling", candidates=[a, b], answers=answers, bank=_bank(), n_sites=2
    )
    assert table.columns["state"][SITE, 35] == asof.STATE_ERA


def test_candidates_from_tables_merge_and_derive_how_an_episode_closed() -> None:
    bocpd_rows = pl.DataFrame(
        {"site": [0, 0], "onset": [10, 40], "alarm": [11, 41], "end": [40, 55]}
    )
    stat_rows = pl.DataFrame(
        {"site": [0, 0], "onset": [12, 100], "alarm": [20, 101], "end": [30, 110]}
    )
    union, low = asof.candidates_from_tables(bocpd_rows=bocpd_rows, stat_rows=stat_rows)
    assert low == []
    assert [(c.detector, c.alarm, c.ended_by) for c in union] == [
        ("bocpd", 11, "superseded"),
        ("bocpd", 41, "return"),
        ("stat", 101, "return"),
    ]
    assert union[0].stat_alarm == 20
    assert union[1].stat_alarm is None


def test_register_windows_and_the_ecr_candidates() -> None:
    rows = (
        prompts.RegisterRow(0, 100, 2.0, "solar", "under 1 km", True),
        prompts.RegisterRow(1, 100, 2.0, "solar", "listed under a neighbouring substation", False),
        prompts.RegisterRow(0, 300, 0.7, "solar", "under 1 km", False),
    )
    windows = asof.watch_windows(rows=rows)
    assert windows == [(0, 114, 220)]
    low = [
        _candidate(detector="stat3", episode_id="stat30", alarm=50),
        _candidate(detector="stat3", episode_id="stat31", alarm=150),
        _candidate(detector="stat3", episode_id="stat32", alarm=180),
    ]
    assert [c.alarm for c in asof.ecr_candidates(windows=windows, low_threshold=low)] == [150]
    union = [_candidate(alarm=150, episode_id="stat5")]
    both = asof.llm_ecr_candidates(union=union, windows=windows, low_threshold=low)
    assert [c.alarm for c in both] == [150, 180]


def test_column_names_and_the_written_table_match_experiment_709(tmp_path: Path) -> None:
    asof.ensure_709_importable()
    signal = pytest.importorskip("e00709_signal")
    assert asof.COLUMN_NAMES == signal.COLUMN_NAMES
    assert asof.INT_COLUMNS == signal.INT_COLUMNS
    assert (asof.STATE_NONE, asof.STATE_UNKNOWN, asof.STATE_ERA, asof.STATE_SWITCH) == (
        0, 1, signal.STATE_ERA, signal.STATE_SWITCH,
    )  # fmt: skip
    c = _candidate()
    table = asof.build_table(
        arm="det", candidates=[c], answers=asof.det_answers(candidates=[c]), bank=_bank(), n_sites=2
    )
    asof.write_table(table=table, folder=tmp_path)
    read = signal.read_signal(arm="det", folder=tmp_path)
    assert read.causal
    for name in asof.COLUMN_NAMES:
        assert np.allclose(read.columns[name], table.columns[name], atol=1e-5), name
    assert read.episodes.rows() == table.episodes.rows()
    block = signal.era_block(
        signal=read,
        site=np.array([SITE, SITE]),
        origin_day=np.array([30, 31]),
        cut_day=10,
        target_std=np.ones(2),
        tau_of_day=np.arange(N_DAYS, dtype=float),
    )
    assert block[0, signal.COL_ACTIVE] == 0.0
    assert block[1, signal.COL_ACTIVE] == 1.0
    held = signal.hold_mask(signal=read, cut_day=60)
    assert held[SITE, 28:30].all()
    assert not held[SITE, 30:].any()
    assert not held[1].any()


def test_the_episode_end_is_the_detectors_own_end_so_hold_mask_reads_no_later_answer() -> None:
    asof.ensure_709_importable()
    signal = pytest.importorskip("e00709_signal")
    n_days = 200
    ramp = np.tile((np.arange(n_days) * 0.1)[:, None], (1, features.N_FEATURES))
    bank = asof.DetectorBank(
        n_days=n_days,
        spread=np.ones((2, features.N_FEATURES)),
        delta={(SITE, "stat"): ramp},
        vectors={(SITE, "stat", 30): (57, np.full(12, 5.0))},
    )
    era = _candidate()
    late = _candidate(episode_id="stat1", onset=60, alarm=62, end=190, ended_by="return")
    answers = {
        era.key: [asof.Resolved(kind="alarm", day=30, label="era", confidence=0.9)],
        late.key: [asof.Resolved(kind="alarm", day=62, label=None, confidence=0.0)],
    }
    table = asof.build_table(arm="t", candidates=[era, late], answers=answers, bank=bank, n_sites=2)
    assert table.episodes["end"].to_list() == [45, 190]
    read = signal.Signal(
        arm="t", columns=table.columns, episodes=table.episodes, causal=True,
    )  # fmt: skip
    held = signal.hold_mask(signal=read, cut_day=150)
    assert np.flatnonzero(held[SITE]).tolist() == [28, 29, *range(60, 150)]


# ---------------------------------------------------------------- the gates


def test_the_go_token_must_equal_the_first_line_of_the_go_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(label, "STAGE1_GO_FILE", tmp_path / "go.txt")
    assert not label.go_token_valid(token="a" * 20)
    (tmp_path / "go.txt").write_text("a" * 20 + "\nsecond line\n")
    assert label.go_token_valid(token="a" * 20)
    assert not label.go_token_valid(token="b" * 20)
    assert not label.go_token_valid(token="a" * 5)
    assert not label.go_token_valid(token=None)


def test_the_pin_must_match_the_prompt_code(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(label, "PIN_PATH", tmp_path / "pin.json")
    assert not label.pin_matches()
    (tmp_path / "pin.json").write_text(json.dumps({"prompt_version_sha256": "0" * 64}))
    assert not label.pin_matches()
    (tmp_path / "pin.json").write_text(
        json.dumps({"prompt_version_sha256": label.prompt_version_hash()})
    )
    assert label.pin_matches()


def test_the_pin_needs_a_passing_probe_on_the_current_prompt_code(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(label, "PROBE_RESULT_PATH", tmp_path / "probe_result.json")
    assert label.probe_blocker() is not None
    current = label.prompt_version_hash()
    for passed, version, ok in (
        (True, current, True),
        (False, current, False),
        (None, current, False),
        (True, "0" * 64, False),
    ):
        (tmp_path / "probe_result.json").write_text(
            json.dumps({"passed": passed, "prompt_version_sha256": version})
        )
        assert (label.probe_blocker() is None) is ok


def test_limits_above_the_stage_1_maximum_are_refused() -> None:
    parser = label._parser()
    ok = parser.parse_args(
        ["--max-usd", "17", "--max-calls", "480", "--max-input-tokens", "3800000"]
    )
    assert label._limits_from(args=ok) is not None
    big = parser.parse_args(
        ["--max-usd", "50", "--max-calls", "480", "--max-input-tokens", "3800000"]
    )
    assert label._limits_from(args=big) is None
    assert label._limits_from(args=parser.parse_args([])) is None


def test_resume_rebuilds_finished_checks_from_a_log() -> None:
    c = _candidate()
    good = {
        "call_id": f"llm:{c.key}_alarm", "tag": "label", "status": "ok", "check_offset_days": 0,
        "answer": {"label": "era", "confidence": 0.9, "onset_offset_days": -2},
    }  # fmt: skip
    bad = {
        "call_id": f"llm:{c.key}_plus7", "tag": "label", "status": "ok", "check_offset_days": 7,
        "answer": None,
    }  # fmt: skip
    partial = label.resolved_from_log(entries=[good, bad], candidates=[c], arm="llm")
    assert [(r.kind, r.label) for r in partial[c.key]] == [("alarm", "era")]
    exhausted = label.resolved_from_log(entries=[good, bad, bad], candidates=[c], arm="llm")
    assert [(r.kind, r.label, r.day) for r in exhausted[c.key]] == [
        ("alarm", "era", 30),
        ("plus7", None, 37),
    ]


# ---------------------------------------------------------------- the prompt (real data)


def test_profile_blocks_have_no_partial_week() -> None:
    for span in range(1, 40):
        blocks = prompts.profile_blocks(onset=100, day0=100 + span - 1)
        since = [b for b in blocks if b[0].startswith("since")]
        assert since[0][1] == 100
        assert since[-1][2] == 100 + span - 1
        if span < 7:
            assert len(since) == 1
        else:
            assert all(b[2] - b[1] + 1 >= 7 for b in since)


def test_register_table_shows_only_published_rows_near_the_onset() -> None:
    near = prompts.RegisterRow(0, 20, 8.91, "solar", "under 1 km", True)
    far = prompts.RegisterRow(0, 900, 3.0, "wind", "under 1 km", True)
    other = prompts.RegisterRow(1, 20, 3.0, "wind", "under 1 km", True)
    c = _candidate()
    shown = prompts.shown_register_rows(rows=(near, far, other), candidate=c, day0=40, lag=14)
    assert shown == [near]
    assert prompts.shown_register_rows(rows=(near,), candidate=c, day0=33, lag=14) == []
    assert prompts.shown_register_rows(rows=(near,), candidate=c, day0=34, lag=14) == [near]


@pytest.fixture(scope="module")
def real_inputs() -> prompts.Inputs:
    try:
        return prompts.load_inputs()
    except FileNotFoundError, OSError:
        pytest.skip("the private data folder is not on this machine")


def test_stored_candidates_number_about_ninety_two() -> None:
    try:
        union = asof.load_stored_candidates()
    except FileNotFoundError, OSError:
        pytest.skip("the stored episode tables are not on this machine")
    assert 88 <= len(union) <= 96
    assert all(c.end < 0 or c.end >= c.alarm for c in union)
    assert {c.ended_by for c in union} <= {"return", "superseded", "open"}


def test_a_real_prompt_has_no_calendar_key_and_passes_the_cut(
    real_inputs: prompts.Inputs,
) -> None:
    union = asof.load_stored_candidates()
    candidate = next(c for c in union if features.PRIMARIES[c.site] == "D10" and c.alarm > 1000)
    for kind, day in (("alarm", candidate.alarm), ("plus7", candidate.alarm + 7)):
        check = prompts.Check(candidate=candidate, kind=kind, day=day)
        prompt = prompts.prompt_of(inputs=real_inputs, check=check)
        prompts.assert_cut_changes_nothing(inputs=real_inputs, check=check, prompt=prompt)
        assert not prompts.FORBIDDEN_PATTERN.search(prompt.text)
        assert 'register_row": null' in prompt.text
        assert "TABLE 7" not in prompt.text
    with_register = prompts.RegisterRow(
        candidate.site, candidate.onset - 10, 4.2, "solar", "under 1 km", True
    )
    check = prompts.Check(candidate=candidate, kind="alarm", day=candidate.alarm)
    prompt = prompts.prompt_of(inputs=real_inputs, check=check, register=(with_register,))
    assert "TABLE 7" in prompt.text
    prompts.assert_cut_changes_nothing(
        inputs=real_inputs, check=check, prompt=prompt, register=(with_register,)
    )


# ---------------------------------------------------------------- the call loop, with a stub


class _Fake:
    """Stands in for the command-line tool: scripted replies, no network, no `claude`."""

    def __init__(
        self, replies: list[str], *, limit: bool = False, key_source: str = "none"
    ) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.limit = limit
        self.key_source = key_source

    def __call__(self, *, prompt: object) -> label.trial.RawRun:
        self.calls += 1
        text = self.replies.pop(0) if self.replies else "OK"
        events: list[dict] = [
            {"type": "system", "tools": [], "skills": [], "mcp_servers": [], "cwd": "/empty",
             "apiKeySource": self.key_source},
            {"type": "assistant", "message": {"model": label.trial.MODEL_ID}},
            {"type": "result", "is_error": False, "result": text, "total_cost_usd": 0.03,
             "usage": {"input_tokens": 100, "output_tokens": 20}},
        ]  # fmt: skip
        if self.limit:
            events.insert(
                1,
                {"type": "rate_limit_event",
                 "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour"}},
            )  # fmt: skip
        stdout = "\n".join(json.dumps(e) for e in events) + "\n"
        return label.trial.RawRun(
            stdout=stdout, stderr="", returncode=0, seconds=0.1, directory="/empty"
        )


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.SimpleNamespace]:
    """Point the run at a scratch folder under the private data folder, with cheap prompts."""
    if not label.OUT_DIR.parent.exists():
        pytest.skip("the private data folder is not on this machine")
    scratch = label.OUT_DIR / "test_scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    run_dir = scratch / "stage1"
    monkeypatch.setattr(label, "RUN_DIR", run_dir)
    monkeypatch.setattr(label, "LOG_PATH", run_dir / "call_log.jsonl")
    monkeypatch.setattr(label, "STOP_PATH", run_dir / "stopped.json")
    monkeypatch.setattr(
        label.prompts, "prompt_of", lambda **_: label.Prompt(system="s", text="t", images=())
    )
    monkeypatch.setattr(label.prompts, "assert_cut_changes_nothing", lambda **_: None)
    monkeypatch.setattr(label.trial, "cli_version", lambda: "stub")
    yield types.SimpleNamespace(inputs=types.SimpleNamespace(last_day=200), run_dir=run_dir)
    shutil.rmtree(scratch, ignore_errors=True)


def _run(
    stubbed,
    fake: _Fake,
    monkeypatch,
    *,
    resume: bool = False,
    limits=None,
    expect="none",
    probe_only=False,
    candidates=None,
):
    monkeypatch.setattr(label.trial, "run_claude", fake)
    return label.run_stage1(
        candidates=candidates or [_candidate()],
        inputs=stubbed.inputs,
        register=None,
        limits=limits or label.Limits(10.0, 50, 1_000_000),
        arm="llm",
        resume=resume,
        expect_key_source=expect,
        probe_only=probe_only,
    )


def test_the_run_probes_then_checks_stops_early_and_runs_the_closure_check(
    stubbed, monkeypatch
) -> None:
    fake = _Fake(["OK", _reply(), _reply(), _reply(label="switch")])
    assert _run(stubbed, fake, monkeypatch) == 0
    assert fake.calls == 4
    entries = label.read_log()
    kinds = [label.split_call_id(call_id=e["call_id"])[2] for e in entries if e["tag"] == "label"]
    assert kinds == ["alarm", "plus7", "closure"]
    assert all(e["prompt_version_sha256"] == label.prompt_version_hash() for e in entries)
    resolved = label.resolved_from_log(entries=entries, candidates=[_candidate()], arm="llm")
    assert [(r.kind, r.label) for r in resolved[_candidate().key]] == [
        ("alarm", "era"),
        ("plus7", "era"),
        ("closure", "switch"),
    ]


def test_an_invalid_reply_is_asked_for_once_more_and_a_resume_sends_nothing(
    stubbed, monkeypatch
) -> None:
    fake = _Fake(
        [
            "OK",
            "not json",
            _reply(confidence=0.5),
            _reply(confidence=0.5),
            _reply(confidence=0.5),
            _reply(confidence=0.5),
        ]
    )
    assert _run(stubbed, fake, monkeypatch) == 0
    attempts = [e["attempt"] for e in label.read_log() if e["tag"] == "label"]
    assert attempts[:2] == [0, 1]
    before = fake.calls
    assert _run(stubbed, fake, monkeypatch, resume=True) == 0
    assert fake.calls == before + 1  # only the billing probe, which is sent on every start


def test_a_rate_limit_stops_the_run_at_once_with_a_stop_file(stubbed, monkeypatch) -> None:
    fake = _Fake(["OK"], limit=True)
    assert _run(stubbed, fake, monkeypatch) == 2
    assert fake.calls == 1
    assert label.STOP_PATH.exists()


def test_the_budget_guard_refuses_a_call_past_the_call_limit(stubbed, monkeypatch) -> None:
    fake = _Fake(["OK", _reply(confidence=0.5)])
    assert _run(stubbed, fake, monkeypatch, limits=label.Limits(10.0, 2, 1_000_000)) == 3
    assert fake.calls == 2


def test_a_different_key_source_aborts_before_any_check_call(stubbed, monkeypatch) -> None:
    fake = _Fake(["OK"], key_source="none")
    assert _run(stubbed, fake, monkeypatch, expect="ANTHROPIC_API_KEY") == 4
    assert fake.calls == 1
    probe = _Fake(["OK"])
    assert _run(stubbed, probe, monkeypatch, resume=True, probe_only=True, expect=None) == 0
    assert probe.calls == 1


def test_an_unexpected_error_sets_the_stop_flag_and_no_call_follows(stubbed, monkeypatch) -> None:
    first, second, third = (
        _candidate(episode_id="stat0"),
        _candidate(episode_id="stat1", onset=60, alarm=62, end=90),
        _candidate(episode_id="stat2", onset=100, alarm=102, end=130),
    )

    def cut_test(**kwargs: prompts.Check) -> None:
        if kwargs["check"].candidate.episode_id == "stat1":
            msg = "a later reading changed the prompt"
            raise ValueError(msg)

    monkeypatch.setattr(label.prompts, "assert_cut_changes_nothing", cut_test)
    monkeypatch.setattr(label, "MAX_WORKERS", 1)
    fake = _Fake(["OK"])
    status = _run(stubbed, fake, monkeypatch, candidates=[first, second, third])
    assert status == 3
    assert "unexpected ValueError" in json.loads(label.STOP_PATH.read_text())["reason"]
    keys = {
        label.split_call_id(call_id=e["call_id"])[1]
        for e in label.read_log()
        if e["tag"] == "label"
    }
    assert keys <= {first.key}
    assert fake.calls == len(label.read_log())


def test_a_call_that_reports_another_key_source_trips_the_run(stubbed, monkeypatch) -> None:
    class Drift(_Fake):
        def __call__(self, *, prompt: object) -> label.trial.RawRun:
            run = super().__call__(prompt=prompt)
            self.key_source = "ANTHROPIC_API_KEY"
            return run

    fake = Drift(["OK"], key_source="none")
    assert _run(stubbed, fake, monkeypatch) == 4
    assert fake.calls == 2
    assert label.STOP_PATH.exists()
    assert label.read_log()[-1]["status"] == "failed"


def test_every_log_entry_has_times_and_each_run_writes_a_record(stubbed, monkeypatch) -> None:
    fake = _Fake(["OK", _reply(), _reply(), _reply(label="switch")])
    assert _run(stubbed, fake, monkeypatch) == 0
    assert all(e["started_utc"] and e["finished_utc"] for e in label.read_log())
    record = json.loads((stubbed.run_dir / "run_1.json").read_text())
    assert record["calls"] == 4
    assert record["finished_utc"] >= record["started_utc"]
    assert {"usd", "input_tokens", "output_tokens", "elapsed_seconds"} <= record.keys()
    assert "seven_day_utilization_first" in record


def test_a_resume_ignores_entries_of_another_prompt_version(stubbed, monkeypatch) -> None:
    fake = _Fake(["OK", _reply(), _reply(), _reply(label="switch")])
    assert _run(stubbed, fake, monkeypatch) == 0
    entries = label.read_log()
    assert label.log_versions(entries=entries) == {label.prompt_version_hash()}
    candidates = [_candidate()]
    assert (
        label.resolved_from_log(entries=entries, candidates=candidates, arm="llm", version="other")
        == {}
    )
    assert label.resolved_from_log(
        entries=entries, candidates=candidates, arm="llm", version=label.prompt_version_hash()
    )
    label.LOG_PATH.write_text(
        "".join(json.dumps(e | {"prompt_version_sha256": "old"}) + "\n" for e in entries)
    )
    before = fake.calls
    assert _run(stubbed, fake, monkeypatch, resume=True) == 0
    assert fake.calls > before + 1
    moved = label.move_log_aside()
    assert moved
    assert not label.LOG_PATH.exists()


def _spend_args(**changes: object) -> types.SimpleNamespace:
    base = {
        "go_token": "t" * 20, "probe_only": False, "expect_key_source": "none", "resume": True,
        "new_version": False, "max_usd": 1.0, "max_calls": 10, "max_input_tokens": 1000,
        "arm": "llm", "primary": ["D8"],
    }  # fmt: skip
    return types.SimpleNamespace(**(base | changes))


def test_spend_refuses_a_resume_over_a_log_of_another_prompt_version(
    v2: None, stubbed, monkeypatch
) -> None:
    label.RUN_DIR.mkdir(parents=True, exist_ok=True)
    label.LOG_PATH.write_text(json.dumps({"prompt_version_sha256": "old", "call_id": "x"}) + "\n")
    monkeypatch.setattr(label, "go_token_valid", lambda **_: True)
    monkeypatch.setattr(label, "pin_matches", lambda: True)
    args = _spend_args()
    assert label._spend(args=args, candidates=[], low=[]) == 1
    args.resume = False
    assert label._spend(args=args, candidates=[], low=[]) == 1


def test_primary_restricts_the_candidates_and_refuses_an_unknown_name() -> None:
    other = _candidate(site=1, episode_id="stat9")
    both = [_candidate(), other]
    only = label.restrict_primaries(candidates=both, primaries=[features.PRIMARIES[1]])
    assert [c.key for c in only] == [other.key]
    assert len(label.restrict_primaries(candidates=both, primaries=None)) == 2
    with pytest.raises(ValueError, match="not a Primary"):
        label.restrict_primaries(candidates=both, primaries=["X99"])


def _known_tables() -> pl.DataFrame:
    """Return a synthetic era table: one dated era and one dated row with no era index."""
    first, second = features.PRIMARIES[0], features.PRIMARIES[1]
    return pl.DataFrame(
        {
            "primary": [first, second],
            "era_index": [0, None],
            "register_date": [features.day_of(index=26), None],
            "telemetry_date": [features.day_of(index=29), features.day_of(index=95)],
        },
        schema_overrides={"era_index": pl.Int32},
    )


def test_known_answers_name_eras_unresolved_rows_and_both_sites_of_an_event() -> None:
    from datetime import timedelta

    era = _known_tables()
    first, second = features.PRIMARIES[0], features.PRIMARIES[1]
    start = features.GRID_START
    events = pl.DataFrame(
        {
            "source": ["experiment 580 verified", "experiment 41 labelled set"],
            "site": [second, first],
            "other_site": [first, None],
            "onset_utc": [start + timedelta(days=200), start + timedelta(days=300)],
            "end_utc": [start + timedelta(days=205), start + timedelta(days=301)],
        }
    )
    era_candidate = _candidate()
    unresolved = _candidate(site=1, episode_id="stat1", onset=96, alarm=98)
    via_other = _candidate(episode_id="stat2", onset=201, alarm=203, end=-1)
    labelled = _candidate(episode_id="stat3", onset=301, alarm=303, end=-1)
    nothing = _candidate(episode_id="stat4", onset=500, alarm=503, end=-1)
    cands = [era_candidate, unresolved, via_other, labelled, nothing]
    known = asof.known_answers(candidates=cands, era_table=era, known_events=events)
    assert known[era_candidate.key].kind == "era"
    assert known[unresolved.key].kind == "unresolved"
    assert (known[via_other.key].kind, known[via_other.key].via) == ("switch", "other_site")
    assert known[labelled.key].source == "experiment 41 labelled set"
    assert known[nothing.key].kind == "none"
    wide = asof.true_classes(candidates=cands, era_table=era, known_events=events)
    strict = asof.true_classes(
        candidates=cands, era_table=era, known_events=events, use_other_site=False
    )
    assert wide[via_other.key] == "switch"
    assert strict[via_other.key] == "noise"
    assert wide[labelled.key] == "noise"  # `ceiling` reads only the verified events
    assert label.ceiling_changes(candidates=cands, era_table=era, known_events=events) == 1


def test_the_agreement_report_prints_counts_a_confusion_table_and_no_other_date(capsys) -> None:
    from datetime import timedelta

    era = _known_tables()
    events = pl.DataFrame(
        {
            "source": ["experiment 580 verified"],
            "site": [features.PRIMARIES[1]],
            "other_site": [None],
            "onset_utc": [features.GRID_START + timedelta(days=200)],
            "end_utc": [features.GRID_START + timedelta(days=205)],
        },
        schema_overrides={"other_site": pl.String},
    )
    c = _candidate()
    entries = [
        {
            "tag": "label", "call_id": f"llm:{c.key}_{kind}", "status": "ok",
            "check_offset_days": offset,
            "answer": {"label": "era", "confidence": 0.9, "onset_offset_days": -2},
        }
        for kind, offset in (("alarm", 0), ("plus7", 7))
    ]  # fmt: skip
    label.print_agreement(
        arm="llm",
        candidates=[c],
        entries=entries,
        era_table=era,
        known_events=events,
        n_days=N_DAYS,
    )
    out = capsys.readouterr().out
    assert "Confusion table, check kind alarm" in out
    assert "n < 5: no interval" in out
    assert "Delays for each known era" in out
    assert str(features.day_of(index=26)) not in out or features.PRIMARIES[0] == "D8"


# ---------------------------------------------------------------- the calendar-free prompt


def test_the_local_hour_grid_follows_the_london_clock() -> None:
    hours = prompts.local_hour_grid()
    assert hours.size == features.N_HALF_HOURS
    assert hours[0] == 1  # 00:00 UTC on 1 September 2019 is 01:00 British Summer Time
    assert hours[features.day_index(day=features.day_of(index=91)) * 48] == 0  # 1 December, GMT
    assert set(np.unique(hours)) == set(range(24))


def test_the_prompt_features_read_nothing_after_the_check_day() -> None:
    rng = np.random.default_rng(0)
    power = rng.uniform(0.1, 0.9, features.N_HALF_HOURS)
    day0 = 800
    cut = power[: (day0 + 1) * 48]
    full = prompts.prompt_features(power_site=power, day0=day0)
    short = prompts.prompt_features(power_site=cut, day0=day0)
    assert np.isfinite(full[day0 - 30 : day0 + 1]).all()
    assert np.array_equal(full, short, equal_nan=True)


def test_one_climatology_serves_every_check_day_of_a_calendar_month() -> None:
    rng = np.random.default_rng(1)
    power = rng.uniform(0.1, 0.9, features.N_HALF_HOURS)
    first_of_month = features.day_index(day=features.day_of(index=800).replace(day=1))
    early = prompts.prompt_features(power_site=power, day0=first_of_month + 3)
    late = prompts.prompt_features(power_site=power, day0=first_of_month + 20)
    rows = slice(first_of_month - 40, first_of_month + 4)
    assert np.array_equal(early[rows], late[rows], equal_nan=True)


def test_a_real_prompt_has_no_weekday_split(real_inputs: prompts.Inputs) -> None:
    candidate = next(
        c for c in asof.load_stored_candidates() if features.PRIMARIES[c.site] == "D10"
    )
    check = prompts.Check(candidate=candidate, kind="alarm", day=candidate.alarm)
    text = prompts.prompt_of(inputs=real_inputs, check=check).text.lower()
    assert "weekday" not in text
    assert "weekend" not in text


# ---------------------------------------------------------------- attempt 2

V1_HASH = "4910761a8dcb3bb98de7a31b05cb08b19f1a5c029c2a66c56012a74220019dc5"


@pytest.fixture
def v2() -> Iterator[None]:
    label.configure_attempt(name="v2")
    yield
    label.configure_attempt(name="v1")


def test_attempt_1_is_unchanged_and_reproducible() -> None:
    label.configure_attempt(name="v1")
    assert label.attempt_name() == "v1"
    assert label.prompt_version_hash() == V1_HASH
    assert label.PIN_PATH.name == "prompt_pin.json"
    assert label.RUN_DIR.name == "stage1"
    assert label.STAGE1_GO_FILE.name == "stage1_go_token.txt"
    assert (prompts.SYSTEM_PROMPT, prompts.SCHEMA_NO_REGISTER, prompts.SCHEMA_WITH_REGISTER) == (
        label.V1_PROMPT_TEXTS
    )
    assert "60 words" in prompts.SCHEMA_NO_REGISTER


def test_attempt_2_has_a_new_identity_and_its_own_files(v2: None) -> None:
    assert label.prompt_version_hash() != V1_HASH
    assert label.PIN_PATH.name == "prompt_pin_v2.json"
    assert label.RUN_DIR.name == "stage1_v2"
    assert label.LOG_PATH.parent.name == "stage1_v2"
    assert label.STAGE1_GO_FILE.name == "stage1_go_token_v2.txt"
    assert label.PROBE_PROMPT_DIR.name == "prompts_v2"
    assert label.PROBE_RESULT_PATH.name == "probe_result_v2.json"
    assert "40 words" in prompts.SCHEMA_NO_REGISTER
    assert "no code fence" in prompts.SYSTEM_PROMPT
    assert 'has no "register_row" key' in prompts.SCHEMA_NO_REGISTER
    assert "always include this key" in prompts.SCHEMA_WITH_REGISTER
    label.configure_attempt(name="v1")
    assert label.prompt_version_hash() == V1_HASH


def test_the_v2_schema_has_no_register_row_line_for_arm_llm(v2: None) -> None:
    lines = [line.strip() for line in prompts.SCHEMA_NO_REGISTER.splitlines()]
    assert not any(line.startswith('"register_row"') for line in lines)


def test_a_reply_without_register_row_parses_under_the_v2_schema(v2: None) -> None:
    body = json.loads(_reply())
    del body["register_row"]
    assert label.parse_reply(text=json.dumps(body)).register_row is None


def test_the_pin_refuses_without_a_passing_v2_probe_on_the_current_hash(
    v2: None, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(label, "PROBE_RESULT_PATH", tmp_path / "probe_result_v2.json")
    assert label.probe_blocker() is not None
    result = {"prompt_version_sha256": label.prompt_version_hash(), "passed": True}
    label.PROBE_RESULT_PATH.write_text(json.dumps(result))
    assert label.probe_blocker() is None
    label.PROBE_RESULT_PATH.write_text(json.dumps(result | {"prompt_version_sha256": V1_HASH}))
    assert "different prompt version" in str(label.probe_blocker())
    label.PROBE_RESULT_PATH.write_text(json.dumps(result | {"passed": False}))
    assert label.probe_blocker() is not None


def _write_token(folder: Path, name: str, token: str) -> Path:
    path = folder / name
    path.write_text(token + "\n")
    return path


def test_a_claimed_token_is_refused_again_even_when_the_file_is_rewritten(tmp_path: Path) -> None:
    path = _write_token(tmp_path, "stage1_go_token_v2.txt", "a" * 20)
    assert label.claim_token(token="a" * 20, path=path) is not None
    assert not path.exists()
    assert label.claim_token(token="a" * 20, path=path) is None
    _write_token(tmp_path, "stage1_go_token_v2.txt", "b" * 20)
    assert label.claim_token(token="b" * 20, path=path) is not None
    _write_token(tmp_path, "stage1_go_token_v2.txt", "b" * 20)  # B again, after two uses
    assert not label.token_valid(token="b" * 20, path=path)
    _write_token(tmp_path, "stage1_go_token_v2.txt", "a" * 20)
    assert not label.token_valid(token="a" * 20, path=path)
    _write_token(tmp_path, "stage1_go_token_v2.txt", "c" * 20)
    assert label.token_valid(token="c" * 20, path=path)
    assert len(list(tmp_path.glob("*.used.*"))) == 2


def test_a_probe_token_cannot_start_the_labelling(tmp_path: Path) -> None:
    probe_file = _write_token(tmp_path, "probe_go_token_v2.txt", "p" * 20)
    assert label.claim_token(token="p" * 20, path=probe_file) is not None
    stage_file = _write_token(tmp_path, "stage1_go_token_v2.txt", "p" * 20)
    assert not label.token_valid(token="p" * 20, path=stage_file)
    assert label.claim_token(token="p" * 20, path=stage_file) is None


def test_two_claims_of_one_token_cannot_both_succeed(tmp_path: Path) -> None:
    path = _write_token(tmp_path, "stage1_go_token_v2.txt", "q" * 20)
    first = label.claim_token(token="q" * 20, path=path)
    second = label.claim_token(token="q" * 20, path=path)
    assert first is not None
    assert second is None


def test_a_v2_run_claims_the_token_before_the_first_call_and_a_resume_needs_a_new_one(
    v2: None, stubbed, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(label, "STAGE1_GO_FILE", tmp_path / "stage1_go_token_v2.txt")
    _write_token(tmp_path, "stage1_go_token_v2.txt", "g" * 20)
    monkeypatch.setattr(label, "pin_matches", lambda: True)
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "era.parquet")
    monkeypatch.setattr(label.asof, "ERA_TABLE_PATH", tmp_path / "era.parquet")
    monkeypatch.setattr(label, "arm_inputs", lambda *, candidates, **_: (candidates, None))
    monkeypatch.setattr(label.prompts, "load_inputs", lambda: stubbed.inputs)
    seen: list[bool] = []

    class Spy(_Fake):
        def __call__(self, *, prompt: object) -> label.trial.RawRun:
            seen.append((tmp_path / "stage1_go_token_v2.txt").exists())
            return super().__call__(prompt=prompt)

    monkeypatch.setattr(label.trial, "run_claude", Spy(["OK", _reply(), _reply(), _reply()]))
    args = _spend_args(
        go_token="g" * 20, resume=False, max_usd=4.0, max_calls=100, max_input_tokens=800_000
    )
    assert label._spend(args=args, candidates=[_candidate()], low=[]) == 0
    assert seen
    assert not any(seen)  # the file was already renamed when the first call started
    assert list(tmp_path.glob("stage1_go_token_v2.txt.used.*"))
    args.resume = True
    assert label._spend(args=args, candidates=[_candidate()], low=[]) == 1


def test_attempt_1_is_closed_and_attempt_2_is_limited_to_d8_arm_llm(v2: None, monkeypatch) -> None:
    monkeypatch.setattr(label, "go_token_valid", lambda **_: True)
    monkeypatch.setattr(label, "pin_matches", lambda: True)
    assert label._spend(args=_spend_args(primary=None), candidates=[], low=[]) == 1
    assert label._spend(args=_spend_args(primary=["D10"]), candidates=[], low=[]) == 1
    assert label._spend(args=_spend_args(primary=["D8", "D10"]), candidates=[], low=[]) == 1
    assert label._spend(args=_spend_args(arm="llm-ecr"), candidates=[], low=[]) == 1
    ok = {"max_usd": 4.0, "max_calls": 100, "max_input_tokens": 800_000}
    too_much = {"max_usd": 4.5, "max_calls": 141, "max_input_tokens": 1_200_000}
    for name, value in too_much.items():
        assert label._limits_from(args=_spend_args(**(ok | {name: value}))) is None
    limits = {"max_usd": 4.0, "max_calls": 140, "max_input_tokens": 1_100_000}
    assert label._limits_from(args=_spend_args(**limits)) is not None
    label.configure_attempt(name="v1")
    assert label._spend(args=_spend_args(), candidates=[], low=[]) == 1


def test_attempt_2_shares_no_path_with_attempt_1_and_refuses_reparse(monkeypatch, capsys) -> None:
    names = (
        "RUN_DIR", "PROMPT_DIR", "LOG_PATH", "STOP_PATH", "REPARSED_LOG_PATH", "SIGNALS_DIR",
        "STAGE1_GO_FILE", "PIN_PATH", "PROBE_PROMPT_DIR", "PROBE_RESULT_PATH",
    )  # fmt: skip
    label.configure_attempt(name="v1")
    first = {n: getattr(label, n) for n in names}
    label.configure_attempt(name="v2")
    try:
        second = {n: getattr(label, n) for n in names}
        assert all(first[n] != second[n] for n in names)
        monkeypatch.setattr(sys, "argv", ["e00712_label.py", "--attempt", "v2", "--reparse"])
        assert label.main() == 1
        assert "nothing to re-parse" in capsys.readouterr().err
    finally:
        label.configure_attempt(name="v1")


def test_the_format_counts_read_structure_only() -> None:
    def entry(text: str, call_id: str, attempt: int = 0) -> dict:
        return {"tag": "label", "status": "ok", "call_id": call_id, "attempt": attempt,
                "raw_output": text}  # fmt: skip

    clean = json.loads(_reply())
    del clean["register_row"]
    entries = [
        entry(json.dumps(clean), "llm:a_alarm"),
        entry("```json\n" + json.dumps(clean) + "\n```", "llm:b_alarm"),
        entry(json.dumps(clean), "llm:b_alarm", 1),
        entry(json.dumps({**clean, "register_row": None}), "llm:c_alarm"),
        entry(json.dumps({**clean, "reason": "word " * 41}), "llm:d_alarm"),
        entry("not json", "llm:e_alarm"),
        entry("still not json", "llm:e_alarm", 1),
    ]
    counts = label.reply_format_counts(entries=entries)
    assert counts["replies"] == 7
    assert counts["not_bare_json"] == 3
    assert counts["register_row_present"] == 1
    assert counts["reason_over_40_words"] == 1
    assert counts["fail_lenient_parser"] == 2
    assert (counts["v2_first_reply_valid"], counts["v2_retry_valid"]) == (2, 1)
    assert counts["v2_never_valid"] == 2
    assert not any(isinstance(v, str) for v in counts.values())


def test_an_unanswered_check_with_one_attempt_is_unfinished_not_never_valid() -> None:
    entries = [{"tag": "label", "call_id": "llm:a_alarm", "attempt": 0, "answer": None}]
    accounting = label.reply_accounting(entries=entries)
    assert accounting["unfinished"] == 1
    assert accounting["never_valid"] == 0


def test_retries_are_counted_by_reason_and_summarised(stubbed, monkeypatch) -> None:
    fake = _Fake(["OK", "not json", _reply(), _reply(), _reply(label="switch")])
    assert _run(stubbed, fake, monkeypatch) == 0
    entries = label.read_log()
    retries = [e for e in entries if e["tag"] == "label" and e["attempt"] == 1]
    assert [e["retry_reason"] for e in retries] == ["parse_failure"]
    assert all(e["retry_reason"] is None for e in entries if e.get("attempt") == 0)
    assert all(e["run_attempt"] == "v1" for e in entries)
    accounting = label.reply_accounting(entries=entries)
    assert accounting["first_reply_valid"] == 2
    assert accounting["retry_valid"] == 1
    assert accounting["never_valid"] == 0
    assert accounting["retries_by_reason"] == {"parse_failure": 1}


def test_a_check_with_two_invalid_replies_is_never_valid(stubbed, monkeypatch) -> None:
    fake = _Fake(["OK", "not json", "still not json", _reply(), _reply()])
    assert _run(stubbed, fake, monkeypatch) == 0
    accounting = label.reply_accounting(entries=label.read_log())
    assert accounting["never_valid"] == 1
    assert accounting["first_reply_valid"] == 2
    attempts = [e["attempt"] for e in label.read_log() if e["tag"] == "label"][:2]
    assert attempts == [0, 1]  # one retry at most


def test_a_resume_after_a_failed_call_records_the_retry_as_other(stubbed, monkeypatch) -> None:
    label.RUN_DIR.mkdir(parents=True, exist_ok=True)
    first = _candidate()
    failed = {
        "call_id": f"llm:{first.key}_alarm", "attempt": 0, "tag": "label", "status": "failed",
        "prompt_version_sha256": label.prompt_version_hash(), "input_tokens": 0,
        "output_tokens": 0, "cost_usd": 0.0, "check_offset_days": 0, "model": "m",
    }  # fmt: skip
    label.LOG_PATH.write_text(json.dumps(failed) + "\n")
    fake = _Fake(["OK", _reply(), _reply(), _reply()])
    assert _run(stubbed, fake, monkeypatch, resume=True) == 0
    retry = next(e for e in label.read_log() if e.get("attempt") == 1)
    assert retry["retry_reason"] == "other"
