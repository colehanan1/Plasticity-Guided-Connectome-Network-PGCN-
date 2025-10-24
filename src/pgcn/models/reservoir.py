"""Reservoir module approximating mushroom body sparsity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


BaseModule = nn.Module if nn is not None else object


@dataclass(frozen=True)
class ReservoirConfig:
    n_pn: int = 50
    n_kc: int = 2000
    n_mbon: int = 10
    kc_sparsity: float = 0.05
    connectivity: str = "hemibrain"


class DrosophilaReservoir(BaseModule):
    """Simple PN→KC→MBON reservoir with configurable sparsity."""

    def __init__(
        self,
        n_pn: int = ReservoirConfig.n_pn,
        n_kc: int = ReservoirConfig.n_kc,
        n_mbon: int = ReservoirConfig.n_mbon,
        kc_sparsity: float = ReservoirConfig.kc_sparsity,
        connectivity: str = ReservoirConfig.connectivity,
        cache_dir: Optional[Path | str] = None,
        pn_kc_matrix: Optional[Tensor | np.ndarray | Sequence[Sequence[float]]] = None,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch is required to instantiate DrosophilaReservoir.")
        if not 0.0 < kc_sparsity <= 1.0:
            raise ValueError("kc_sparsity must lie within (0, 1].")

        resolved_cache = Path(cache_dir) if cache_dir is not None else None
        matrix_data = self._coerce_matrix(pn_kc_matrix) if pn_kc_matrix is not None else None
        if matrix_data is None and resolved_cache is not None:
            matrix_data = self._load_connectome_matrix(resolved_cache)

        resolved_n_pn = n_pn
        resolved_n_kc = n_kc
        if matrix_data is not None:
            matrix_kc, matrix_pn = matrix_data.shape
            if (resolved_n_pn, resolved_n_kc) == (
                ReservoirConfig.n_pn,
                ReservoirConfig.n_kc,
            ):
                resolved_n_pn = matrix_pn
                resolved_n_kc = matrix_kc
            elif (resolved_n_pn, resolved_n_kc) != (matrix_pn, matrix_kc):
                raise ValueError(
                    "Provided PN→KC matrix has shape "
                    f"{matrix_data.shape} but reservoir configured for "
                    f"({resolved_n_kc}, {resolved_n_pn})."
                )

        super().__init__()
        self.n_pn = resolved_n_pn
        self.n_kc = resolved_n_kc
        self.n_mbon = n_mbon
        self.kc_sparsity = kc_sparsity
        self.connectivity = connectivity
        self._matrix_data = matrix_data

        self.pn_to_kc = nn.Linear(self.n_pn, self.n_kc, bias=False)
        self.kc_to_mbon = nn.Linear(self.n_kc, n_mbon, bias=True)

        self._initialise_connectivity()

    def _initialise_connectivity(self) -> None:
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            if self._matrix_data is not None:
                weight = torch.as_tensor(self._matrix_data, dtype=self.pn_to_kc.weight.dtype)
                mask = (weight > 0.0).to(weight.dtype)
                weight = self._normalize_weights(weight) * mask
            else:
                weight = torch.randn(self.n_kc, self.n_pn, generator=generator) * 0.1
                mask = torch.rand(
                    self.n_kc,
                    self.n_pn,
                    generator=generator,
                    device=weight.device,
                    dtype=weight.dtype,
                )
                threshold = torch.quantile(mask, 1.0 - self.kc_sparsity)
                mask = (mask >= threshold).float()
                weight = weight * mask
            self._assign_connectivity(weight, mask)
        for param in self.pn_to_kc.parameters():
            param.requires_grad = False

    def forward(self, pn_activity):  # type: ignore[override]
        if pn_activity.dim() == 1:
            pn_activity = pn_activity.unsqueeze(0)
        kc = F.relu(self.pn_to_kc(pn_activity))
        kc = self._enforce_sparsity(kc)
        return F.relu(self.kc_to_mbon(kc))

    def _enforce_sparsity(self, kc_activity: torch.Tensor) -> torch.Tensor:
        if self.kc_sparsity >= 1.0:
            return kc_activity
        k = max(1, int(round(self.n_kc * self.kc_sparsity)))
        _, indices = torch.topk(kc_activity, k=k, dim=-1)
        mask = torch.zeros_like(kc_activity)
        mask.scatter_(-1, indices, 1.0)
        return kc_activity * mask

    def _assign_connectivity(self, weight: Tensor, mask: Tensor) -> None:
        if weight.shape != (self.n_kc, self.n_pn):
            raise ValueError(
                "PN→KC weight matrix must have shape "
                f"({self.n_kc}, {self.n_pn}), received {tuple(weight.shape)}."
            )
        weight = weight.to(self.pn_to_kc.weight.device, dtype=self.pn_to_kc.weight.dtype)
        mask = mask.to(self.pn_to_kc.weight.device, dtype=self.pn_to_kc.weight.dtype)
        if hasattr(self, "pn_kc_mask"):
            self.pn_kc_mask.copy_(mask)
        else:
            self.register_buffer("pn_kc_mask", mask.clone())
        self.pn_to_kc.weight.copy_(weight * mask)

    def _normalize_weights(self, weight: Tensor) -> Tensor:
        weight = weight.clamp_min(0.0)
        row_sums = weight.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        return weight / row_sums

    def _coerce_matrix(
        self, matrix: Tensor | np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        if isinstance(matrix, np.ndarray):
            array = matrix
        elif torch is not None and isinstance(matrix, torch.Tensor):
            array = matrix.detach().cpu().numpy()
        else:
            array = np.asarray(matrix, dtype=float)
        if array.ndim != 2:
            raise ValueError("PN→KC matrix must be two-dimensional.")
        return array.astype(float, copy=False)

    def _load_connectome_matrix(self, cache_dir: Path) -> np.ndarray:
        nodes_path = self._resolve_cache_path(cache_dir, "nodes")
        edges_path = self._resolve_cache_path(cache_dir, "edges")
        nodes = pd.read_parquet(nodes_path)
        edges = pd.read_parquet(edges_path)

        type_column = self._infer_column(nodes.columns, ["type", "cell_type", "node_type"])
        if type_column is None:
            raise ValueError("Connectome cache nodes table missing a node type column.")
        node_id_column = self._infer_column(nodes.columns, ["node_id", "id"])
        if node_id_column is None:
            raise ValueError("Connectome cache nodes table missing a node identifier column.")

        normalized_types = nodes[type_column].astype(str).str.upper()
        nodes = nodes.assign(_pgcn_type=normalized_types)
        pn_nodes = nodes[nodes._pgcn_type == "PN"].sort_values(node_id_column)
        kc_nodes = nodes[nodes._pgcn_type == "KC"].sort_values(node_id_column)
        if pn_nodes.empty or kc_nodes.empty:
            raise ValueError("Connectome cache does not contain PN and KC populations.")

        pn_ids = pn_nodes[node_id_column].to_numpy()
        kc_ids = kc_nodes[node_id_column].to_numpy()
        pn_index = {int(node_id): idx for idx, node_id in enumerate(pn_ids)}
        kc_index = {int(node_id): idx for idx, node_id in enumerate(kc_ids)}

        source_col = self._infer_column(edges.columns, ["source_id", "source", "pre_id"])
        target_col = self._infer_column(edges.columns, ["target_id", "target", "post_id"])
        weight_col = self._infer_column(
            edges.columns,
            ["synapse_weight", "weight", "synapse_count", "count", "size"],
        )
        if source_col is None or target_col is None or weight_col is None:
            raise ValueError("Connectome cache edges table missing required columns.")

        pn_mask = edges[source_col].isin(pn_index)
        kc_mask = edges[target_col].isin(kc_index)
        pn_kc_edges = edges[pn_mask & kc_mask]
        if pn_kc_edges.empty:
            raise ValueError("Connectome cache does not contain PN→KC edges.")

        grouped = (
            pn_kc_edges.groupby([target_col, source_col])[weight_col]
            .sum()
            .reset_index()
        )

        matrix = np.zeros((len(kc_ids), len(pn_ids)), dtype=float)
        for row in grouped.itertuples(index=False):
            target = kc_index[int(getattr(row, target_col))]
            source = pn_index[int(getattr(row, source_col))]
            matrix[target, source] = float(getattr(row, weight_col))

        return matrix

    def _resolve_cache_path(self, cache_dir: Path, stem: str) -> Path:
        canonical = cache_dir / f"{stem}.parquet"
        if canonical.exists():
            return canonical
        candidates = sorted(cache_dir.glob(f"{stem}*.parquet"), key=lambda path: path.stat().st_mtime)
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(
            (
                f"Could not find '{stem}.parquet' in {cache_dir}. "
                f"Run `pgcn-cache --out {cache_dir}` (use `--use-sample-data` if necessary) "
                "before hydrating the reservoir."
            )
        )

    @staticmethod
    def _infer_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
        lowered = {col.lower(): col for col in columns}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        return None


__all__ = ["DrosophilaReservoir", "ReservoirConfig"]
