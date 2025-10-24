"""Tests for the DrosophilaReservoir PN→KC hydration logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:  # pragma: no cover - optional dependency in CI
    import torch
except ImportError:  # pragma: no cover - handled via pytest skip marker
    torch = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="tests require torch")

from pgcn.models.reservoir import DrosophilaReservoir


@pytest.fixture
def sample_matrix() -> np.ndarray:
    # Row sums intentionally non-uniform to exercise normalisation.
    return np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def test_reservoir_accepts_preparsed_matrix(sample_matrix: np.ndarray) -> None:
    reservoir = DrosophilaReservoir(pn_kc_matrix=sample_matrix)

    expected = sample_matrix.copy()
    for idx, row in enumerate(expected):
        row_sum = row.sum()
        if row_sum > 0:
            expected[idx] = row / row_sum
    mask = torch.tensor(sample_matrix > 0, dtype=reservoir.pn_to_kc.weight.dtype)

    weight = reservoir.pn_to_kc.weight.detach().cpu()
    assert weight.shape == sample_matrix.shape
    torch.testing.assert_close(weight, torch.tensor(expected).float())
    torch.testing.assert_close(reservoir.pn_kc_mask.cpu(), mask)
    assert not reservoir.pn_to_kc.weight.requires_grad


def test_reservoir_loads_weights_from_cache(tmp_path: Path) -> None:
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2, 3, 4],
            "type": ["PN", "PN", "KC", "KC"],
        }
    )
    nodes.to_parquet(tmp_path / "nodes.parquet")

    edges = pd.DataFrame(
        {
            "source_id": [1, 2, 1, 2],
            "target_id": [3, 3, 4, 4],
            "synapse_weight": [3.0, 1.0, 0.0, 2.0],
        }
    )
    edges.to_parquet(tmp_path / "edges.parquet")

    reservoir = DrosophilaReservoir(cache_dir=tmp_path)

    weight = reservoir.pn_to_kc.weight.detach().cpu().numpy()
    mask = reservoir.pn_kc_mask.detach().cpu().numpy()

    assert weight.shape == (reservoir.n_kc, reservoir.n_pn)
    assert reservoir.n_pn == 2
    assert reservoir.n_kc == 2

    # The first KC has two incoming edges that should be normalised to sum to 1.
    np.testing.assert_allclose(weight[0].sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(weight[1, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(mask, (weight > 0).astype(mask.dtype))
    assert not reservoir.pn_to_kc.weight.requires_grad
