import json
import numpy as np
import pytest
import statsmodels.api as sm
from scipy import sparse

from src.analysis.sparse_logistic import build_design, cluster_covariance, fit_mle, gradient, hessian, inference, log_likelihood, wald_test


def synthetic_data():
    # Intercept, body-v-wide and T-v-wide; all categories have both outcomes.
    rng = np.random.default_rng(20260325)
    direction = np.tile(np.array([0, 1, 2]), 30)
    probability = np.choose(direction, [0.42, 0.55, 0.66])
    outcome = rng.binomial(1, probability).astype(float)
    design = np.column_stack([np.ones(len(direction)), direction == 1, direction == 2]).astype(float)
    clusters = np.repeat(np.arange(15), 6)
    return sparse.csr_matrix(design), design, outcome, clusters


def test_sparse_mle_and_cluster_covariance_match_statsmodels_glm():
    csr, dense, outcome, clusters = synthetic_data()
    sparse_fit = fit_mle(csr, outcome)
    reference = sm.GLM(outcome, dense, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    robust = cluster_covariance(csr, outcome, sparse_fit["beta"], clusters)
    assert sparse_fit["converged"]
    assert sparse_fit["converged"] == sparse_fit["optimizer_success"]
    np.testing.assert_allclose(sparse_fit["beta"], reference.params, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sparse_fit["log_likelihood"], reference.llf, rtol=1e-7, atol=1e-7)
    # Ambos Hessianos son los de la log-verosimilitud: -X'WX, sin escala extra.
    np.testing.assert_allclose(
        hessian(sparse_fit["beta"], csr),
        reference.model.hessian(reference.params),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(robust, reference.cov_params(), rtol=1e-6, atol=1e-7)
    assert np.linalg.norm(gradient(sparse_fit["beta"], csr, outcome)) < 1e-7
    assert np.all(np.linalg.eigvalsh(-hessian(sparse_fit["beta"], csr)) > 0)
    details = inference(sparse_fit["beta"], robust)
    np.testing.assert_allclose(details["std_error"], reference.bse, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(details["p_value"], reference.pvalues, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(details["ci_low"], reference.params - 1.959963984540054 * reference.bse, rtol=1e-6, atol=1e-7)
    ours = wald_test(sparse_fit["beta"], robust, (1, 2))
    expected = reference.wald_test(np.array([[0, 1, 0], [0, 0, 1]]), scalar=True)
    assert ours["df"] == 2
    np.testing.assert_allclose(ours["statistic"], expected.statistic, rtol=1e-6)
    np.testing.assert_allclose(ours["p_value"], expected.pvalue, rtol=1e-6)


@pytest.mark.parametrize("outcome", [np.zeros(6), np.ones(6)])
def test_rejects_constant_outcome(outcome):
    design = sparse.csr_matrix(np.column_stack([np.ones(6), [0, 1, 0, 1, 0, 1]]))
    with pytest.raises(ValueError, match="variacion"):
        fit_mle(design, outcome)


def test_rejects_singular_design_and_insufficient_clusters():
    design = sparse.csr_matrix(np.ones((6, 2)))
    with pytest.raises(ValueError, match="singular"):
        fit_mle(design, np.array([0, 1, 0, 1, 0, 1]))
    csr, _, outcome, _ = synthetic_data()
    with pytest.raises(ValueError, match="clusters"):
        cluster_covariance(csr, outcome, np.zeros(3), np.zeros(36))


def test_rejects_invalid_covariance_and_missing_wald_terms():
    with pytest.raises(ValueError, match="Covarianza"):
        inference(np.array([0.0, np.nan]), np.eye(2))
    with pytest.raises(ValueError, match="ausentes"):
        wald_test(np.zeros(2), np.eye(2), (2,))
    with pytest.raises(ValueError, match="contraste"):
        wald_test(np.zeros(2), np.array([[1.0, 2.0], [2.0, 4.0]]), (0, 1))


def _category_contract():
    return ({"direction": ("wide", "body", "T"), "surface": ("Hard", "Clay", "Grass"), "period": ("to_2009", "2010s", "2020s"), "server": ("Aaron", "Boris", "Carlos")}, {"direction": "wide", "surface": "Hard", "period": "to_2009", "server": "Aaron"})


def _category_frame():
    import pandas as pd
    return pd.DataFrame({"direction": ["wide", "body", "T"], "surface": ["Hard", "Clay", "Grass"], "period": ["to_2009", "2010s", "2020s"], "server": ["Aaron", "Boris", "Carlos"]})


def test_sparse_design_column_mapping_and_references():
    matrix, mapping = build_design(_category_frame(), *_category_contract())
    expected_terms = ["Intercept", "direction[body]", "direction[T]", "surface[Clay]", "surface[Grass]", "period[2010s]", "period[2020s]", "server[Boris]", "server[Carlos]"]
    assert mapping == {term: index for index, term in enumerate(expected_terms)}
    assert np.array_equal(matrix.toarray(), np.array([[1,0,0,0,0,0,0,0,0],[1,1,0,1,0,1,0,1,0],[1,0,1,0,1,0,1,0,1]]))


def test_sparse_design_uses_coo_coordinates_without_dense_category_columns(monkeypatch):
    import scipy.sparse as sp
    frame = _category_frame()
    calls = []
    original = sp.coo_matrix
    monkeypatch.setattr(sp, "coo_matrix", lambda *args, **kwargs: (calls.append(args[0][1]), original(*args, **kwargs))[1])
    matrix, _ = build_design(frame, *_category_contract())
    assert matrix.format == "csr" and len(calls) == 1
    rows, columns = calls[0]
    assert rows.ndim == columns.ndim == 1
    assert matrix.nnz == 11


@pytest.mark.parametrize("column,value", [("direction", "unknown"), ("surface", "Carpet"), ("period", "future"), ("server", "Unknown")])
def test_sparse_design_rejects_invalid_categories(column, value):
    frame = _category_frame(); frame.loc[0, column] = value
    with pytest.raises(ValueError, match="Categoria"):
        build_design(frame, *_category_contract())


def test_sparse_design_rejects_missing_reference_and_duplicate_categories():
    frame = _category_frame().query("server != 'Aaron'")
    with pytest.raises(ValueError, match="referencia ausente"):
        build_design(frame, *_category_contract())
    categories, references = _category_contract(); categories["direction"] = ("wide", "body", "body")
    with pytest.raises(ValueError, match="duplicada"):
        build_design(_category_frame(), categories, references)


def test_non_convergence_and_excessive_gradient_are_not_publishable(monkeypatch):
    csr, _, outcome, _ = synthetic_data()
    class Result: success=False; x=np.zeros(3); nit=0
    monkeypatch.setattr("src.analysis.sparse_logistic.optimize.minimize", lambda *args, **kwargs: Result())
    result = fit_mle(csr, outcome)
    assert not result["converged"] and not result["publishable"]
    assert result["gradient_norm"] > 1e-6
    assert result["converged"] == result["optimizer_success"]
    assert result["optimizer_status"] is None
    assert result["optimizer_message"] is None
    assert result["objective_value"] is None
    json.dumps(result["diagnostics"], allow_nan=False)


def test_warm_start_uses_exact_coefficients_and_validates_contract(monkeypatch):
    csr, _, outcome, _ = synthetic_data()
    original = __import__("src.analysis.sparse_logistic", fromlist=["optimize"]).optimize.minimize
    observed = []
    def capture(*args, **kwargs):
        observed.append(args[1].copy())
        return original(*args, **kwargs)
    monkeypatch.setattr("src.analysis.sparse_logistic.optimize.minimize", capture)
    initial = np.array([0.1, -0.2, 0.3])
    result = fit_mle(csr, outcome, initial_coefficients=initial, max_iterations=200)
    np.testing.assert_array_equal(observed, [initial])
    assert result["used_warm_start"] and result["initial_point"] == "previous_solution"
    assert result["max_iterations"] == 200
    with pytest.raises(ValueError, match="dimension"):
        fit_mle(csr, outcome, initial_coefficients=np.zeros(2))
    with pytest.raises(ValueError, match="finitos"):
        fit_mle(csr, outcome, initial_coefficients=np.array([0.0, np.nan, 0.0]))


def test_warm_start_continuation_matches_long_fit_without_worsening_likelihood():
    csr, _, outcome, _ = synthetic_data()
    first = fit_mle(csr, outcome, max_iterations=20)
    continued = fit_mle(csr, outcome, initial_coefficients=first["beta"], max_iterations=200)
    long = fit_mle(csr, outcome, max_iterations=220)
    assert continued["log_likelihood"] >= first["log_likelihood"] - 1e-10
    assert continued["gradient_norm"] <= first["gradient_norm"] + 1e-6
    np.testing.assert_allclose(continued["beta"], long["beta"], atol=1e-6, rtol=1e-6)
    assert continued["tolerance"] == first["tolerance"] == long["tolerance"]


def test_perfect_and_near_separation_are_not_silently_publishable():
    for outcome in (np.array([0, 0, 0, 1, 1, 1], dtype=float), np.array([0, 0, 1, 1, 1, 1], dtype=float)):
        design = sparse.csr_matrix(np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]))
        try:
            result = fit_mle(design, outcome)
            assert not result["publishable"] or (result["gradient_norm"] <= 1e-6 and np.max(np.abs(result["beta"])) <= 30)
        except (ValueError, np.linalg.LinAlgError):
            pass


def test_full_sparse_design_is_never_densified(monkeypatch):
    csr, _, outcome, _ = synthetic_data()
    original_toarray = sparse.csr_matrix.toarray
    original_todense = sparse.csr_matrix.todense

    def reject_full_toarray(matrix, *args, **kwargs):
        if matrix.shape == csr.shape:
            raise AssertionError("full design densified with toarray")
        return original_toarray(matrix, *args, **kwargs)

    def reject_full_todense(matrix, *args, **kwargs):
        if matrix.shape == csr.shape:
            raise AssertionError("full design densified with todense")
        return original_todense(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_full_toarray)
    monkeypatch.setattr(sparse.csr_matrix, "todense", reject_full_todense)
    result = fit_mle(csr, outcome)
    assert result["publishable"]
    assert result["converged"]
    assert np.isfinite(result["log_likelihood"])
    assert result["gradient_norm"] <= 1e-6


def test_dense_auxiliary_memory_guard():
    design = sparse.csr_matrix((2, 5000))
    with pytest.raises(MemoryError):
        fit_mle(design, np.array([0, 1]))
