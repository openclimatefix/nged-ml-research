"""Tests of the tuning extension, the edge check, and gate G3b at two taus (experiment 734)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import e00734_edge as edge
import e00734_hmm as hmm
import e00734_tuning as tuning


def _rho_on_table() -> list[dict]:
    """Objective rising towards the lowest rho_on: 0.349, 0.364, 0.371 (the first tuning)."""
    return [
        {"rho_on": on, "rho_off": off, "objective": obj, "eligible": True}
        for on, obj in ((1e-3, 0.349), (3e-4, 0.364), (1e-4, 0.371))
        for off in (1 / 60, 1 / 240)
    ]


def test_the_extension_adds_eight_settings_with_the_same_rho_off_and_rho_cont() -> None:
    """Test 2 new rho_on values times 2 rho_off times 2 rho_cont, named after the first 12."""
    assert len(tuning.SETTINGS) == 12
    assert len(tuning.SETTINGS_EXT) == 8
    assert {s[0] for s in tuning.SETTINGS_EXT} == {3e-5, 1e-5}
    assert {s[1:] for s in tuning.SETTINGS_EXT} == {s[1:] for s in tuning.SETTINGS}
    assert tuning.ALL_SETTINGS[:12] == tuning.SETTINGS
    assert tuning.ext_names() == [f"own{i}" for i in range(12, 20)]


def test_a_rho_on_still_improving_at_the_lowest_grid_value_is_an_edge_hit() -> None:
    """Test the first tuning's rho_on (0.349, 0.364, 0.371 towards the lowest value) is a hit."""
    table = _rho_on_table()
    got = edge.parameter_edge(
        arm="H-own",
        name="rho_on",
        grid=[r["rho_on"] for r in table],
        chosen=1e-4,
        objective_by_value=edge.profile(table=table, key="rho_on"),
    )
    assert got["hit"] is True
    assert got["edge"] == "lowest"
    assert got["position"] == "1 of 3"
    assert got["inward_neighbour"] == 3e-4
    assert abs(got["gain_towards_edge"] - 0.007) < 1e-9


def test_a_plateau_at_an_edge_and_an_interior_choice_are_not_hits() -> None:
    """Test a gain of 0.002 at the edge is a plateau, and a middle value has no edge."""
    plateau = edge.parameter_edge(
        arm="H-own",
        name="rho_off",
        grid=[1 / 240, 1 / 60],
        chosen=1 / 60,
        objective_by_value={1 / 240: 0.369, 1 / 60: 0.371},
    )
    assert plateau["edge"] == "highest"
    assert plateau["hit"] is False
    assert plateau["note"] == "plateau"
    interior = edge.parameter_edge(
        arm="H-nbr",
        name="kappa",
        grid=[0, 5, 20],
        chosen=5,
        objective_by_value={0: 0.432, 5: 0.448, 20: 0.43},
    )
    assert interior["edge"] is None
    assert interior["hit"] is False


def test_an_ineligible_inward_neighbour_is_reported_and_is_not_a_hit() -> None:
    """Test a NaN neighbour (ineligible) gives no gain and the note says why."""
    got = edge.parameter_edge(
        arm="PELT",
        name="penalty",
        grid=[2.0, 20.0, 200.0],
        chosen=200.0,
        objective_by_value={2.0: 0.5, 20.0: float("nan"), 200.0: 0.3},
    )
    assert got["hit"] is False
    assert got["note"] == "inward neighbour ineligible"


def test_tau_on_the_grid_floor_is_a_hit_unless_a_called_rate_binds() -> None:
    """Test tau 0.5 with the big-call rate equal to the limit is rate-bound, not a hit."""
    taus = (0.5, 0.6, 0.7, 0.8)
    bound = edge.tau_edge(
        arm="H-nbr",
        grid=taus,
        chosen=0.5,
        big_rate=42 / 26.33,
        big_limit=42 / 26.33,
        all_rate=3.19,
        all_limit=6.76,
    )
    assert bound["edge"] == "lowest"
    assert bound["rate_bound"] is True
    assert bound["hit"] is False
    free = edge.tau_edge(
        arm="H-nbr",
        grid=taus,
        chosen=0.5,
        big_rate=1.2,
        big_limit=1.595,
        all_rate=3.19,
        all_limit=6.76,
    )
    assert free["hit"] is True
    mid = edge.tau_edge(
        arm="H-own",
        grid=taus,
        chosen=0.7,
        big_rate=1.2,
        big_limit=1.595,
        all_rate=3.19,
        all_limit=6.76,
    )
    assert mid["edge"] is None
    assert mid["hit"] is False


def test_the_summary_says_edge_hit_and_that_choose_does_not_extend() -> None:
    """Test the printed line names the hit and says that `choose` never extends a grid."""
    table = _rho_on_table()
    check = edge.parameter_edge(
        arm="H-own",
        name="rho_on",
        grid=[r["rho_on"] for r in table],
        chosen=1e-4,
        objective_by_value=edge.profile(table=table, key="rho_on"),
    )
    (line,) = edge.summary_lines(checks=[check])
    assert "EDGE HIT" in line
    assert "never extends a grid by itself" in line


def test_the_profile_takes_the_best_eligible_objective_of_each_value() -> None:
    """Test an ineligible row never raises the profile, and a value with none is NaN."""
    table = [
        {"k": 1.0, "objective": 0.9, "eligible": False},
        {"k": 1.0, "objective": 0.4, "eligible": True},
        {"k": 1.0, "objective": 0.5, "eligible": True},
        {"k": 2.0, "objective": 0.8, "eligible": False},
    ]
    got = edge.profile(table=table, key="k")
    assert got[1.0] == 0.5
    assert np.isnan(got[2.0])


def test_the_first_outputs_are_copied_to_v1_once_and_never_replaced(tmp_path: Path) -> None:
    """Test `*_v1` copies exist after the backup, and a second backup keeps the first copy."""
    (tmp_path / "choice_own.json").write_text('{"a": 1}')
    (tmp_path / "params.sha256").write_text("abc  params.json\n")
    assert sorted(tuning.backup_v1(out=tmp_path)) == ["choice_own_v1.json", "params_v1.sha256"]
    assert (tmp_path / "choice_own_v1.json").read_text() == '{"a": 1}'
    (tmp_path / "choice_own.json").write_text('{"a": 2}')
    assert tuning.backup_v1(out=tmp_path) == []
    assert (tmp_path / "choice_own_v1.json").read_text() == '{"a": 1}'


def _choice(tmp: Path, name: str, on: float) -> None:
    obj = {"h_own": {"rho_on": on, "rho_off": 1 / 60, "rho_cont": 0.2, "tau": 0.8}}
    (tmp / name).write_text(json.dumps(obj))


def test_old_nbr_outputs_are_reused_only_when_the_chosen_h_own_values_are_unchanged(
    tmp_path: Path,
) -> None:
    """Test the tag is `nbr` with no v1 or equal values, and `nbr2` when rho_on changed."""
    assert tuning.nbr_tag(out=tmp_path) == "nbr"
    _choice(tmp_path, "choice_own_v1.json", 1e-4)
    _choice(tmp_path, "choice_own.json", 1e-4)
    assert tuning.nbr_tag(out=tmp_path) == "nbr"
    _choice(tmp_path, "choice_own.json", 3e-5)
    assert tuning.nbr_tag(out=tmp_path) == "nbr2"


def test_the_extension_is_complete_only_when_every_world_and_the_real_record_have_files(
    tmp_path: Path,
) -> None:
    """Test `extension_complete` needs each of the 8 settings for each key."""
    keys = ["734000", "real"]
    (tmp_path / "tune").mkdir()
    for name in tuning.ext_names():
        for key in keys:
            (tmp_path / "tune" / f"{name}_{key}.parquet").write_bytes(b"x")
    assert tuning.extension_complete(out=tmp_path, keys=keys)
    (tmp_path / "tune" / "own19_real.parquet").unlink()
    assert not tuning.extension_complete(out=tmp_path, keys=keys)


def test_gate_g3b_reads_each_arm_at_its_own_tau_on_the_same_synthetic_noise() -> None:
    """Test both arms are reported, each at its own tau, and a lower tau calls at least as much."""
    priors = hmm.Priors(rho_on=1e-4, rho_off=1 / 60, rho_cont=0.2)
    got = tuning.g3b_rates(arms={"H-own": (priors, 0.8), "H-nbr": (priors, 0.5)}, years=0.3, seed=1)
    assert set(got) == {"H-own", "H-nbr"}
    assert got["H-own"]["tau"] == 0.8
    assert got["H-nbr"]["tau"] == 0.5
    assert got["H-nbr"]["called_per_primary_year"] >= got["H-own"]["called_per_primary_year"]
    assert all(isinstance(v["passed"], bool) for v in got.values())


def test_the_second_extension_gives_every_parameter_three_or_more_values_with_the_best_inside() -> (
    None
):
    """Test at most 15 settings, under 8 CPU-hours, with the best so far in the interior."""
    assert len(tuning.SETTINGS_EXT2) <= 15
    assert (
        tuning.projected_cpu_hours(n_settings=len(tuning.SETTINGS_EXT2)) < tuning.EXT2_CAP_CPU_HOURS
    )
    assert len(set(tuning.ALL_SETTINGS)) == len(tuning.ALL_SETTINGS)
    best = (1e-5, 1 / 60, 0.2)
    for k in range(3):
        values = sorted({s[k] for s in tuning.ALL_SETTINGS})
        assert len(values) > 2
        assert values[0] < best[k] < values[-1]
    assert max(s[1] for s in tuning.ALL_SETTINGS) < 1.0
    assert max(s[2] for s in tuning.ALL_SETTINGS) < 1.0
    names = tuning.ext2_names()
    assert names[0] == "own20"
    assert len(names) == len(tuning.SETTINGS_EXT2)
    assert not set(names) & set(tuning.ext_names())


def test_every_new_setting_gives_a_valid_transition_matrix() -> None:
    """Test rows of the joint transition are non-negative and sum to 1 at the new extreme values."""
    for on, off, cont in tuning.SETTINGS_EXT2:
        full = hmm._base(
            p=hmm.Priors(rho_on=on, rho_off=off, rho_cont=cont), grid=hmm.FULL_GRID
        ).full
        assert full.min() >= -1e-12
        np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-9)


def test_the_h_own_grid_is_extended_twice_and_a_third_extension_is_refused(tmp_path: Path) -> None:
    """Test the counter: 1 then 2 pass, a resume passes, 3 and a skipped number raise."""
    with pytest.raises(ValueError, match="comes first"):
        tuning.register_h_own_extension(out=tmp_path, number=2)
    tuning.register_h_own_extension(out=tmp_path, number=1)
    tuning.register_h_own_extension(out=tmp_path, number=2)
    tuning.register_h_own_extension(out=tmp_path, number=2)
    assert tuning.read_state(out=tmp_path)["h_own"] == 2
    with pytest.raises(ValueError, match="at most"):
        tuning.register_h_own_extension(out=tmp_path, number=3)
    assert tuning.read_state(out=tmp_path)["h_own"] == 2


def test_the_first_extension_is_inferred_from_its_files_when_no_state_file_exists(
    tmp_path: Path,
) -> None:
    """Test files of `own12` count as extension 1, so extension 2 may follow at once."""
    (tmp_path / "tune").mkdir()
    (tmp_path / "tune" / "own12_734000.parquet").write_bytes(b"x")
    assert tuning.read_state(out=tmp_path)["h_own"] == 1
    tuning.register_h_own_extension(out=tmp_path, number=2)
    assert tuning.read_state(out=tmp_path)["h_own"] == 2


def test_the_second_backup_adds_the_edge_check_and_never_replaces_a_copy(tmp_path: Path) -> None:
    """Test `*_v2` copies include `edge_check.json`, and a second call keeps the first copies."""
    (tmp_path / "edge_check.json").write_text("1")
    (tmp_path / "choice_own.json").write_text("1")
    assert sorted(tuning.backup_v2(out=tmp_path)) == ["choice_own_v2.json", "edge_check_v2.json"]
    (tmp_path / "edge_check.json").write_text("2")
    assert tuning.backup_v2(out=tmp_path) == []
    assert (tmp_path / "edge_check_v2.json").read_text() == "1"


def test_the_grid_level_rises_with_each_complete_extension(tmp_path: Path) -> None:
    """Test levels 0, 1, 2 as the files of the first and then the second extension appear."""
    keys = ["734000", "real"]
    (tmp_path / "tune").mkdir()
    assert tuning.grid_level(out=tmp_path, keys=keys) == 0
    for names in (tuning.ext_names(), tuning.ext2_names()):
        for name in names:
            for key in keys:
                (tmp_path / "tune" / f"{name}_{key}.parquet").write_bytes(b"x")
        level = tuning.grid_level(out=tmp_path, keys=keys)
    assert level == 2
    (tmp_path / "tune" / "own33_real.parquet").unlink()
    assert tuning.grid_level(out=tmp_path, keys=keys) == 1


def test_the_nbr_tag_tells_three_choices_apart(tmp_path: Path) -> None:
    """Test the tags `nbr`, `nbr2`, and `nbr3` for the first, second, and third choices."""
    _choice(tmp_path, "choice_own_v1.json", 1e-4)
    _choice(tmp_path, "choice_own.json", 1e-5)
    assert tuning.nbr_tag(out=tmp_path) == "nbr2"
    _choice(tmp_path, "choice_own_v2.json", 1e-5)
    _choice(tmp_path, "choice_own.json", 1e-5)
    assert tuning.nbr_tag(out=tmp_path) == "nbr2"
    _choice(tmp_path, "choice_own.json", 3e-6)
    assert tuning.nbr_tag(out=tmp_path) == "nbr3"
    _choice(tmp_path, "choice_own.json", 1e-4)
    assert tuning.nbr_tag(out=tmp_path) == "nbr"


def test_extension_values_continue_the_grid_beyond_the_edge_where_the_choice_sits() -> None:
    """Test PELT extends beyond its high or low edge, kappa upward, and interior values refuse."""
    pelt = tuple(float(v) for v in np.geomspace(2.0, 200.0, 12))
    side, high = tuning.extension_values(grid=pelt, chosen=pelt[-1])
    assert side == "high"
    assert len(high) == 3
    assert high[0] > 200.0
    side, low = tuning.extension_values(grid=pelt, chosen=pelt[0])
    assert side == "low"
    assert low[0] < 2.0
    with pytest.raises(ValueError, match="interior"):
        tuning.extension_values(grid=pelt, chosen=pelt[8])
    kappas = (0.0, 5.0, 20.0)
    assert tuning.extension_values(grid=kappas, chosen=20.0) == ("high", [40.0, 80.0, 160.0])
    with pytest.raises(ValueError, match="natural lower limit"):
        tuning.extension_values(grid=kappas, chosen=0.0, floor=0.0)


def test_the_pelt_and_kappa_grids_are_extended_once_and_with_at_most_three_values(
    tmp_path: Path,
) -> None:
    """Test a repeat with the same values passes, other values and more than 3 values raise."""
    got = tuning.register_other_extension(out=tmp_path, arm="pelt", side="low", values=[1.0, 0.5])
    assert got == [1.0, 0.5]
    assert tuning.register_other_extension(out=tmp_path, arm="pelt", side="low", values=[1.0, 0.5])
    with pytest.raises(ValueError, match="once already"):
        tuning.register_other_extension(out=tmp_path, arm="pelt", side="low", values=[0.3])
    with pytest.raises(ValueError, match="at most 3"):
        tuning.register_other_extension(
            out=tmp_path, arm="kappa", side="high", values=[40.0, 80.0, 160.0, 320.0]
        )
    assert tuning.read_state(out=tmp_path)["kappa"] is None


def test_a_rate_binding_value_is_told_apart_from_an_edge_and_asks_for_a_decision() -> None:
    """Test an interior PELT value next to an ineligible one is rate-binding, a hit asks."""
    nan = float("nan")
    penalties = [1.0, 2.0, 3.0, 4.0]
    objective = {1.0: nan, 2.0: 0.4, 3.0: 0.39, 4.0: 0.3}
    bound = edge.parameter_edge(
        arm="PELT", name="penalty", grid=penalties, chosen=2.0, objective_by_value=objective
    )
    assert bound["edge"] is None
    assert bound["hit"] is False
    assert bound["ineligible_neighbours"] == ["lower"]
    assert bound["classification"].startswith("rate-binding")
    assert edge.equal_effort_lines(checks=[bound]) == []
    hit = edge.parameter_edge(
        arm="H-nbr",
        name="kappa",
        grid=[0, 5, 20],
        chosen=20,
        objective_by_value={0: 0.40, 5: 0.43, 20: 0.46},
    )
    (line,) = edge.equal_effort_lines(checks=[hit])
    assert line.startswith("EQUAL EFFORT: PELT/H-nbr edge hit, needs a decision")


def test_a_hit_after_the_last_extension_is_reported_as_tuned_to_the_edge() -> None:
    """Test the summary line of a final-extension hit says the grid is not extended again."""
    check = edge.parameter_edge(
        arm="H-own",
        name="rho_on",
        grid=[1e-6, 3e-6, 1e-5],
        chosen=1e-6,
        objective_by_value={1e-6: 0.40, 3e-6: 0.38, 1e-5: 0.37},
    ) | {"final_extension": True}
    (line,) = edge.summary_lines(checks=[check])
    assert "tuned to the edge of the extended grid" in line


def test_lower_taus_keeps_the_candidates_within_the_rate_limits() -> None:
    """Test the new taus are those below the floor whose called rates stay within the limits."""
    grid = (0.5, 0.6, 0.9)
    assert tuning.lower_taus(grid=grid, chosen=0.5, within_limits=lambda _t: True) == [
        0.4,
        0.3,
        0.2,
    ]
    assert tuning.lower_taus(grid=grid, chosen=0.5, within_limits=lambda t: t >= 0.3) == [0.4, 0.3]


def test_lower_taus_refuses_an_interior_choice_and_a_rate_bound_floor() -> None:
    """Test the tau extension refuses a non-floor choice and a rate-bound floor."""
    grid = (0.5, 0.6, 0.9)
    with pytest.raises(ValueError, match="not the floor"):
        tuning.lower_taus(grid=grid, chosen=0.6, within_limits=lambda _t: True)
    with pytest.raises(ValueError, match="binds"):
        tuning.lower_taus(grid=grid, chosen=0.5, within_limits=lambda _t: False)


def test_the_tau_extension_is_registered_once_with_the_same_values(tmp_path: Path) -> None:
    """Test a second tau extension with different values is refused."""
    tuning.register_other_extension(out=tmp_path, arm="tau", side="low", values=[0.4, 0.3])
    tuning.register_other_extension(out=tmp_path, arm="tau", side="low", values=[0.4, 0.3])
    with pytest.raises(ValueError, match="tau"):
        tuning.register_other_extension(out=tmp_path, arm="tau", side="low", values=[0.4, 0.2])
    assert tuning.read_state(out=tmp_path)["tau"] is not None
