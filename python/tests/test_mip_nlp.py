import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import ObjectiveSense, SolveResult, _DisjunctiveConstraint
from discopt.solvers import MILPResult, SolveStatus


def _binary_model(name="mip_nlp_route"):
    m = dm.Model(name)
    x = m.binary("x")
    m.minimize(x)
    return m


def _gdp_model(name="mip_nlp_gdp_route"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.either_or([[x <= 3], [x >= 7]], name="mode")
    return m


def _has_disjunctions(model):
    return any(isinstance(c, _DisjunctiveConstraint) for c in model._constraints)


def test_model_solve_routes_mip_nlp_options(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls["model"] = model
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    with pytest.warns(
        UserWarning,
        match="MIP-NLP solver ignores solve_model options: skip_convex_check",
    ):
        result = _binary_model().solve(
            solver="mip-nlp",
            mip_nlp_method="ecp",
            equality_relaxation=True,
            skip_convex_check=True,
        )

    assert result.status == "optimal"
    assert calls["method"] == "ecp"
    assert calls["equality_relaxation"] is True


def test_model_solve_routes_issue12_mip_nlp_options(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    result = _binary_model("issue12_route").solve(
        solver="mip-nlp",
        equality_relaxation=True,
        add_slack=True,
        oa_penalty_factor=7.0,
        max_slack=5.0,
        heuristic_nonconvex=True,
    )

    assert result.status == "optimal"
    assert calls["method"] == "oa"
    assert calls["equality_relaxation"] is True
    assert calls["add_slack"] is True
    assert calls["oa_penalty_factor"] == 7.0
    assert calls["max_slack"] == 5.0
    assert calls["heuristic_nonconvex"] is True


def test_gdp_method_oa_deprecated_alias_routes_to_mip_nlp(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls.update(kwargs)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    with pytest.deprecated_call(match="gdp_method='oa' is deprecated"):
        result = _binary_model("oa_alias").solve(
            gdp_method="oa",
            equality_relaxation=True,
            skip_convex_check=True,
        )

    assert result.status == "optimal"
    assert calls["method"] == "oa"
    assert calls["equality_relaxation"] is True


def test_mip_nlp_and_deprecated_oa_alias_reformulate_gdp(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    captured = []

    def fake_solve_mip_nlp(model, **kwargs):
        captured.append(model)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    _gdp_model("mip_nlp_gdp").solve(solver="mip-nlp")
    with pytest.deprecated_call(match="gdp_method='oa' is deprecated"):
        _gdp_model("oa_alias_gdp").solve(gdp_method="oa")

    assert len(captured) == 2
    assert not _has_disjunctions(captured[0])
    assert not _has_disjunctions(captured[1])


def test_mip_nlp_reserved_methods_raise():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(NotImplementedError, match="mip_nlp_method='fp'"):
        solve_mip_nlp(_binary_model("fp_reserved"), method="fp")


def test_master_slack_rows_and_penalty(monkeypatch):
    from discopt.solvers import lp_backend
    from discopt.solvers.oa import _solve_master_milp

    inspected = {}

    def fake_get_milp_solver():
        def fake_solve_milp(**kwargs):
            inspected.update(kwargs)
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([0.0, 0.0]),
                objective=0.0,
                bound=0.0,
                gap=0.0,
            )

        return fake_solve_milp

    monkeypatch.setattr(lp_backend, "get_milp_solver", fake_get_milp_solver)

    result = _solve_master_milp(
        linear_A_rows=[],
        linear_b_rows=[],
        linear_senses=[],
        oa_A_rows=[np.array([1.0])],
        oa_b_rows=[2.0],
        n_vars=1,
        integrality=np.array([0], dtype=np.int32),
        lb=np.array([0.0]),
        ub=np.array([10.0]),
        obj_coeffs=(np.array([1.0]), 0.0),
        obj_is_linear=True,
        objective_bound_valid=True,
        time_limit=1.0,
        gap_tolerance=1e-4,
        oa_slack_flags=[True],
        oa_penalty_factor=7.0,
        max_slack=5.0,
    )

    assert result.status == SolveStatus.OPTIMAL
    np.testing.assert_allclose(inspected["A_ub"], np.array([[1.0, -1.0]]))
    np.testing.assert_allclose(inspected["b_ub"], np.array([2.0]))
    np.testing.assert_allclose(inspected["c"], np.array([1.0, 7.0]))
    assert inspected["bounds"] == [(0.0, 10.0), (0.0, 5.0)]
    np.testing.assert_array_equal(inspected["integrality"], np.array([0, 0], dtype=np.int32))


def test_equality_relaxation_orientation_uses_objective_sense():
    from discopt.solvers.oa import _equality_relaxation_sigma

    assert _equality_relaxation_sigma(0, np.array([2.0]), ObjectiveSense.MINIMIZE) == -1.0
    assert _equality_relaxation_sigma(0, np.array([2.0]), ObjectiveSense.MAXIMIZE) == 1.0
    assert _equality_relaxation_sigma(0, np.array([0.0]), ObjectiveSense.MINIMIZE) == 1.0
    assert _equality_relaxation_sigma(2, np.array([2.0]), ObjectiveSense.MINIMIZE) == 1.0


def test_heuristic_nonconvex_oa_result_is_uncertified():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    result = solve_mip_nlp(
        _binary_model("heuristic_nonconvex_uncertified"),
        method="oa",
        mip_nlp_options={"heuristic_nonconvex": True},
        time_limit=10,
    )

    assert result.status == "feasible"
    assert result.bound is None
    assert result.gap is None
    assert result.gap_certified is False


def test_mip_nlp_rejects_unsupported_oa_options():
    from discopt.solvers.mip_nlp import solve_mip_nlp

    with pytest.raises(ValueError, match="Unsupported MIP-NLP OA/ECP option"):
        solve_mip_nlp(
            _binary_model("unsupported_oa_option"),
            method="oa",
            mip_nlp_options={"solution_pool": True},
        )
