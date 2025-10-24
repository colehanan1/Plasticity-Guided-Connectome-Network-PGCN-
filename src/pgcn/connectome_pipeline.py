"""Utilities for building a FlyWire-derived connectome cache.

The module exposes :class:`ConnectomePipeline` and a CLI entry-point. The
pipeline is responsible for:

* authenticating against FlyWire via :class:`caveclient.CAVEclient`
* pinning and recording the materialization version used for queries
* discovering synapse and cell tables programmatically
* selecting PN/KC/MBON/DAN populations from the materialization tables
* querying synaptic edges for PN→KC, KC→MBON, and DAN→(KC|MBON) motifs
* estimating node coordinates from synapse centroids
* writing the resulting node/edge tables and metadata to disk

When external connectivity is unavailable the pipeline can fabricate a
self-consistent sample cache so unit tests remain hermetic.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from tenacity import retry, stop_after_attempt, wait_exponential

import requests

try:  # pragma: no cover - optional dependency resolved at runtime
    from caveclient import CAVEclient  # type: ignore
except ImportError:  # pragma: no cover - exercised when caveclient missing
    CAVEclient = None  # type: ignore

from .flywire_access import diagnose_flywire_access


console = Console()

DEFAULT_DATASTACK = "flywire_fafb_production"
DEFAULT_MATERIALIZATION_VERSION = 783
DEFAULT_TOKEN_PATH = Path.home() / ".cloudvolume/secrets/cave-secret.json"
CACHE_FILENAMES = {
    "nodes": "nodes.parquet",
    "edges": "edges.parquet",
    "dan_edges": "dan_edges.parquet",
    "meta": "meta.json",
}

NODE_TYPES = {"PN", "KC", "MBON", "DAN"}


@dataclass(slots=True)
class TableDiscovery:
    """Container for the materialized tables used in the cache."""

    synapse_table: str
    cell_tables: List[str]


@dataclass(slots=True)
class CacheArtifacts:
    """Paths for the generated cache files."""

    nodes: Path
    edges: Path
    dan_edges: Path
    meta: Path


class PipelineError(RuntimeError):
    """Raised when required external resources are missing."""


class ConnectomePipeline:
    """Pipeline to extract and cache FlyWire subgraphs."""

    def __init__(
        self,
        datastack: str = DEFAULT_DATASTACK,
        materialization_version: Optional[int] = DEFAULT_MATERIALIZATION_VERSION,
        cache_dir: Path | str = Path("data") / "cache",
        token_path: Path | str = DEFAULT_TOKEN_PATH,
        chunk_size: int = 512,
    ) -> None:
        self.datastack = datastack
        self.requested_mv = materialization_version
        self.cache_dir = Path(cache_dir)
        self.token_path = Path(token_path)
        self.chunk_size = chunk_size
        self._client: Optional[CAVEclient] = None  # type: ignore[assignment]
        self._node_position_cache: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, use_sample_data: bool = False) -> CacheArtifacts:
        """Execute the pipeline.

        Parameters
        ----------
        use_sample_data:
            When ``True`` a deterministic mock cache is produced instead of
            contacting the FlyWire services. This is invaluable for CI where
            authentication may be unavailable.
        """

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._node_position_cache.clear()

        if use_sample_data:
            self.logger.warning("Generating deterministic sample cache data.")
            return self._write_sample_cache()

        self._preflight_access_check()
        client = self._init_client()
        mv = self._resolve_materialization_version(client)
        tables = self._discover_tables(client, mv)

        node_frames = self._collect_node_tables(client, tables.cell_tables, mv)
        pn_nodes, kc_nodes, mbon_nodes, dan_nodes = self._classify_nodes(node_frames)

        pn_kc_edges, kc_mbon_edges, dan_edges = self._collect_edges(
            client=client,
            mv=mv,
            synapse_table=tables.synapse_table,
            pn_ids=pn_nodes["node_id"].tolist(),
            kc_ids=kc_nodes["node_id"].tolist(),
            mbon_ids=mbon_nodes["node_id"].tolist(),
            dan_ids=dan_nodes["node_id"].tolist(),
        )

        nodes = self._assemble_nodes(
            pn_nodes=pn_nodes,
            kc_nodes=kc_nodes,
            mbon_nodes=mbon_nodes,
            dan_nodes=dan_nodes,
            pn_kc_edges=pn_kc_edges,
            kc_mbon_edges=kc_mbon_edges,
            dan_edges=dan_edges,
        )
        core_edges = pd.concat([pn_kc_edges, kc_mbon_edges], ignore_index=True)

        artifacts = self._write_outputs(nodes, core_edges, dan_edges, mv, tables)
        self.logger.info("Cache generation completed successfully.")
        return artifacts

    # ------------------------------------------------------------------
    # Client utilities
    # ------------------------------------------------------------------
    def _init_client(self) -> CAVEclient:
        if CAVEclient is None:  # pragma: no cover - depends on external install
            raise PipelineError(
                "caveclient is not installed. Install it via `pip install caveclient` "
                "or create the provided Conda environment."
            )

        token = self._read_token()
        console.log(f"Authenticating against datastack '{self.datastack}'.")
        try:
            client = CAVEclient(self.datastack, auth_token=token)
        except requests.HTTPError as exc:  # pragma: no cover - requires live service
            response = exc.response
            status = response.status_code if response is not None else None
            if status == 403:
                raise PipelineError(
                    "FlyWire account lacks 'view' permission for the requested dataset. "
                    "Confirm that your token has been granted access to the datastack or "
                    "rerun `pgcn-cache` with `--use-sample-data` to proceed offline."
                ) from exc
            raise PipelineError(
                "Failed to contact FlyWire services. Verify your VPN, token permissions, "
                "and datastack name before retrying."
            ) from exc
        self._client = client
        return client

    def _preflight_access_check(self) -> None:
        status = diagnose_flywire_access(
            self.datastack,
            extra_token_paths=[self.token_path],
        )

        if status.success:
            dataset = status.dataset or "unknown"
            summary = f"Verified access to {self.datastack} (dataset={dataset})."
            if status.versions_ok and status.versions_count is not None:
                summary += f" {status.versions_count} materialization versions detected."
            console.log(summary)
            return

        if status.token_error:
            raise PipelineError(
                "FlyWire token unavailable. Provide credentials via `pgcn-auth --token` and rerun, "
                "or fall back to `pgcn-cache --use-sample-data` for offline operation."
            )

        if status.info_error:
            if "HTTP 403" in status.info_error:
                raise PipelineError(
                    "FlyWire token is valid but lacks 'view' permission for the FAFB dataset. "
                    "Confirm you generated the token under the authorised FlyWire account, "
                    "store it in ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json, and "
                    "request FAFB access before rerunning."
                )
            if "HTTP 401" in status.info_error:
                raise PipelineError(
                    "FlyWire rejected the provided token (HTTP 401). Refresh the secret via `pgcn-auth` "
                    "or regenerate it at https://global.daf-apis.com/auth/api/v1/user/create_token before retrying."
                )
            raise PipelineError(
                "Failed to contact FlyWire services during preflight access check. "
                f"Details: {status.info_error}"
            )

        raise PipelineError(
            "FlyWire preflight check failed for an unknown reason. Inspect `pgcn-access-check` output "
            "for additional detail before retrying."
        )

    def _read_token(self) -> str:
        try:
            secret_data = json.loads(self.token_path.read_text())
        except FileNotFoundError as exc:  # pragma: no cover - depends on env
            raise PipelineError(
                "Authentication secret not found at "
                f"{self.token_path}. Provide a valid FlyWire CAVE token or "
                "rerun `pgcn-cache` with `--use-sample-data` for offline testing."
            ) from exc

        token = secret_data.get("token")
        if not token:  # pragma: no cover - depends on env
            raise PipelineError(
                f"Authentication token missing in {self.token_path}. Expected a 'token' "
                "field with a FlyWire CAVE token string."
            )
        return token

    def _resolve_materialization_version(self, client: CAVEclient) -> int:
        versions: Sequence[int] = client.materialize.get_versions()  # type: ignore[assignment]
        if not versions:
            raise PipelineError("No materialization versions available for datastack.")

        if self.requested_mv and self.requested_mv in versions:
            mv = self.requested_mv
        else:
            mv = max(versions)
            if self.requested_mv and self.requested_mv not in versions:
                self.logger.warning(
                    "Requested materialization version %s unavailable; falling back to latest %s.",
                    self.requested_mv,
                    mv,
                )
        self.logger.info("Using materialization version %s", mv)
        return mv

    # ------------------------------------------------------------------
    # Table discovery
    # ------------------------------------------------------------------
    def _discover_tables(self, client: CAVEclient, mv: int) -> TableDiscovery:
        tables: Sequence[str] = client.materialize.get_tables(materialization_version=mv)
        if not tables:
            raise PipelineError("No materialization tables discovered.")

        synapse_table_candidates: List[str] = []
        cell_tables: List[str] = []
        for table_name in tables:
            metadata = client.materialize.get_table_metadata(
                table_name, materialization_version=mv
            )
            columns = _extract_column_names(metadata)
            lower_columns = {c.lower() for c in columns}
            if {"pre_pt_root_id", "post_pt_root_id"}.issubset(lower_columns):
                synapse_table_candidates.append(table_name)
            if "root_id" in lower_columns:
                cell_tables.append(table_name)

        if not synapse_table_candidates:
            raise PipelineError("Failed to locate a synapse table with pre/post root ids.")

        synapse_table = synapse_table_candidates[0]
        self.logger.info(
            "Synapse table selected: %s (candidates: %s)",
            synapse_table,
            ", ".join(synapse_table_candidates),
        )

        if not cell_tables:
            raise PipelineError("No cell metadata tables found with a 'root_id' column.")

        self.logger.info("Discovered %d candidate cell tables.", len(cell_tables))
        return TableDiscovery(synapse_table=synapse_table, cell_tables=cell_tables)

    # ------------------------------------------------------------------
    # Node discovery and classification
    # ------------------------------------------------------------------
    def _collect_node_tables(
        self, client: CAVEclient, table_names: Sequence[str], mv: int
    ) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        for table in table_names:
            metadata = client.materialize.get_table_metadata(table, materialization_version=mv)
            columns = _extract_column_names(metadata)
            select_columns = [c for c in columns if any(keyword in c.lower() for keyword in ["root", "type", "class", "glomerulus"])]
            if not select_columns:
                continue
            df = client.materialize.query_table(
                table,
                materialization_version=mv,
                select_columns=select_columns,
            )
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            df.columns = [col.lower() for col in df.columns]
            df["source_table"] = table
            frames.append(df)
        if not frames:
            raise PipelineError("Failed to retrieve cell metadata tables with relevant columns.")
        return frames

    def _classify_nodes(
        self, node_frames: Sequence[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        merged = pd.concat(node_frames, ignore_index=True)
        if "root_id" not in merged.columns:
            raise PipelineError("Cell tables do not provide a 'root_id' column after normalization.")
        merged = merged.dropna(subset=["root_id"])
        merged["root_id"] = merged["root_id"].astype(np.int64)

        type_columns = [c for c in merged.columns if any(k in c for k in ["type", "class"])]
        glomerulus_col = next((c for c in merged.columns if "glomerulus" in c), None)

        records: List[MutableMapping[str, Any]] = []
        for _, row in merged.iterrows():
            inferred_type = _infer_cell_type({col: row.get(col) for col in type_columns})
            if inferred_type not in NODE_TYPES:
                continue
            records.append(
                {
                    "node_id": int(row["root_id"]),
                    "type": inferred_type,
                    "glomerulus": row.get(glomerulus_col) if glomerulus_col else None,
                    "source_table": row.get("source_table"),
                }
            )

        if not records:
            raise PipelineError("Unable to classify any PN/KC/MBON/DAN entries from cell tables.")

        nodes = pd.DataFrame.from_records(records)
        pn_nodes = nodes[nodes["type"] == "PN"].drop_duplicates(subset=["node_id"])
        kc_nodes = nodes[nodes["type"] == "KC"].drop_duplicates(subset=["node_id"])
        mbon_nodes = nodes[nodes["type"] == "MBON"].drop_duplicates(subset=["node_id"])
        dan_nodes = nodes[nodes["type"] == "DAN"].drop_duplicates(subset=["node_id"])

        for name, frame in {
            "PN": pn_nodes,
            "KC": kc_nodes,
            "MBON": mbon_nodes,
            "DAN": dan_nodes,
        }.items():
            if frame.empty:
                raise PipelineError(f"No {name} nodes were classified; inspect cell table heuristics.")

        return pn_nodes, kc_nodes, mbon_nodes, dan_nodes

    # ------------------------------------------------------------------
    # Edge collection
    # ------------------------------------------------------------------
    def _collect_edges(
        self,
        client: CAVEclient,
        mv: int,
        synapse_table: str,
        pn_ids: Sequence[int],
        kc_ids: Sequence[int],
        mbon_ids: Sequence[int],
        dan_ids: Sequence[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pn_kc_edges = self._query_edges(client, mv, synapse_table, pn_ids, kc_ids)
        kc_mbon_edges = self._query_edges(client, mv, synapse_table, kc_ids, mbon_ids)
        dan_kc_edges = self._query_edges(client, mv, synapse_table, dan_ids, kc_ids)
        dan_mbon_edges = self._query_edges(client, mv, synapse_table, dan_ids, mbon_ids)
        dan_edges = pd.concat([dan_kc_edges, dan_mbon_edges], ignore_index=True)
        return pn_kc_edges, kc_mbon_edges, dan_edges

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    def _query_edges(
        self,
        client: CAVEclient,
        mv: int,
        table: str,
        pre_ids: Sequence[int],
        post_ids: Sequence[int],
    ) -> pd.DataFrame:
        if not pre_ids or not post_ids:
            return pd.DataFrame(columns=["source_id", "target_id", "synapse_weight"])

        results: List[pd.DataFrame] = []
        post_set = set(post_ids)
        for offset in range(0, len(pre_ids), self.chunk_size):
            chunk = pre_ids[offset : offset + self.chunk_size]
            df = client.materialize.query_table(
                table,
                materialization_version=mv,
                filter_in_dict={"pre_pt_root_id": chunk},
            )
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df[df["post_pt_root_id"].isin(post_set)]
            if df.empty:
                continue
            weight_col = _resolve_weight_column(df.columns)
            if weight_col not in df.columns:
                raise PipelineError(
                    f"Synapse table {table} is missing a weight/count column among {df.columns.tolist()}"
                )
            df = df.assign(
                source_id=df["pre_pt_root_id"].astype(np.int64),
                target_id=df["post_pt_root_id"].astype(np.int64),
                synapse_weight=df[weight_col].astype(float),
            )

            if "ctr_pt_position" in df.columns:
                for src, tgt, coords in df[["source_id", "target_id", "ctr_pt_position"]].itertuples(index=False):
                    tuples = _normalize_positions(coords)
                    for coord in tuples:
                        self._node_position_cache[int(src)].append(coord)
                        self._node_position_cache[int(tgt)].append(coord)
            if "pre_pt_position" in df.columns:
                for src, coords in df[["source_id", "pre_pt_position"]].itertuples(index=False):
                    tuples = _normalize_positions(coords)
                    for coord in tuples:
                        self._node_position_cache[int(src)].append(coord)
            if "post_pt_position" in df.columns:
                for tgt, coords in df[["target_id", "post_pt_position"]].itertuples(index=False):
                    tuples = _normalize_positions(coords)
                    for coord in tuples:
                        self._node_position_cache[int(tgt)].append(coord)
            results.append(df[["source_id", "target_id", "synapse_weight", *_position_columns(df.columns) ]])

        if not results:
            return pd.DataFrame(columns=["source_id", "target_id", "synapse_weight"])

        combined = pd.concat(results, ignore_index=True)
        grouped = (
            combined.groupby(["source_id", "target_id"], as_index=False)["synapse_weight"].sum()
        )
        grouped = grouped[grouped["synapse_weight"] > 0]
        return grouped

    # ------------------------------------------------------------------
    # Node aggregation
    # ------------------------------------------------------------------
    def _assemble_nodes(
        self,
        pn_nodes: pd.DataFrame,
        kc_nodes: pd.DataFrame,
        mbon_nodes: pd.DataFrame,
        dan_nodes: pd.DataFrame,
        pn_kc_edges: pd.DataFrame,
        kc_mbon_edges: pd.DataFrame,
        dan_edges: pd.DataFrame,
    ) -> pd.DataFrame:
        node_frames = [pn_nodes, kc_nodes, mbon_nodes, dan_nodes]
        nodes = pd.concat(node_frames, ignore_index=True)
        nodes = nodes.drop_duplicates(subset=["node_id"]).reset_index(drop=True)
        if "source_table" in nodes.columns:
            nodes = nodes.drop(columns=["source_table"])

        edge_frames = [pn_kc_edges, kc_mbon_edges, dan_edges]
        all_edges = pd.concat(edge_frames, ignore_index=True)
        synapse_counts = (
            pd.concat(
                [
                    all_edges.groupby("source_id")["synapse_weight"].sum().rename("out_weight"),
                    all_edges.groupby("target_id")["synapse_weight"].sum().rename("in_weight"),
                ],
                axis=1,
            )
            .fillna(0.0)
        )
        nodes = nodes.merge(
            synapse_counts.sum(axis=1).rename("synapse_count"),
            how="left",
            left_on="node_id",
            right_index=True,
        )
        nodes["synapse_count"] = nodes["synapse_count"].fillna(0).astype(float)

        centroid_lookup = self._estimate_centroids()
        coords = nodes["node_id"].map(centroid_lookup)
        coord_tuples = coords.apply(
            lambda value: value if isinstance(value, tuple) else (np.nan, np.nan, np.nan)
        )
        nodes[["x", "y", "z"]] = pd.DataFrame(coord_tuples.tolist(), index=nodes.index)
        nodes[["x", "y", "z"]] = nodes[["x", "y", "z"]].fillna(0.0)
        return nodes

    def _estimate_centroids(self) -> Mapping[int, Tuple[float, float, float]]:
        centroids: Dict[int, Tuple[float, float, float]] = {}
        for node_id, coord_list in self._node_position_cache.items():
            arr = np.asarray(coord_list, dtype=float)
            if arr.size == 0:
                continue
            centroid = tuple(arr.mean(axis=0).tolist())
            centroids[node_id] = centroid  # nm coordinates assumed
        return centroids

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _write_outputs(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        dan_edges: pd.DataFrame,
        mv: int,
        tables: TableDiscovery,
    ) -> CacheArtifacts:
        nodes_path = self.cache_dir / CACHE_FILENAMES["nodes"]
        edges_path = self.cache_dir / CACHE_FILENAMES["edges"]
        dan_edges_path = self.cache_dir / CACHE_FILENAMES["dan_edges"]
        meta_path = self.cache_dir / CACHE_FILENAMES["meta"]

        nodes_to_write = nodes.sort_values("node_id").reset_index(drop=True)
        edges_to_write = edges.sort_values(["source_id", "target_id"]).reset_index(drop=True)[
            ["source_id", "target_id", "synapse_weight"]
        ]
        dan_edges_to_write = (
            dan_edges.sort_values(["source_id", "target_id"]).reset_index(drop=True)[
                ["source_id", "target_id", "synapse_weight"]
            ]
        )

        nodes_to_write.to_parquet(nodes_path, index=False)
        edges_to_write.to_parquet(edges_path, index=False)
        dan_edges_to_write.to_parquet(dan_edges_path, index=False)

        meta = {
            "datastack": self.datastack,
            "materialization_version": mv,
            "synapse_table": tables.synapse_table,
            "cell_tables": tables.cell_tables,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

        return CacheArtifacts(nodes=nodes_path, edges=edges_path, dan_edges=dan_edges_path, meta=meta_path)

    # ------------------------------------------------------------------
    # Sample data (CI helper)
    # ------------------------------------------------------------------
    def _write_sample_cache(self) -> CacheArtifacts:
        nodes = pd.DataFrame(
            [
                {"node_id": 1, "type": "PN", "glomerulus": "DA1", "synapse_count": 12.0, "x": 100.0, "y": 200.0, "z": 300.0},
                {"node_id": 2, "type": "PN", "glomerulus": "DL3", "synapse_count": 9.0, "x": 110.0, "y": 210.0, "z": 310.0},
                {"node_id": 3, "type": "KC", "glomerulus": None, "synapse_count": 15.0, "x": 150.0, "y": 240.0, "z": 320.0},
                {"node_id": 4, "type": "KC", "glomerulus": None, "synapse_count": 18.0, "x": 155.0, "y": 250.0, "z": 330.0},
                {"node_id": 5, "type": "MBON", "glomerulus": None, "synapse_count": 21.0, "x": 180.0, "y": 260.0, "z": 340.0},
                {"node_id": 6, "type": "DAN", "glomerulus": None, "synapse_count": 14.0, "x": 190.0, "y": 270.0, "z": 350.0},
            ]
        )
        edges = pd.DataFrame(
            [
                {"source_id": 1, "target_id": 3, "synapse_weight": 5.0},
                {"source_id": 2, "target_id": 4, "synapse_weight": 4.0},
                {"source_id": 3, "target_id": 5, "synapse_weight": 7.0},
                {"source_id": 4, "target_id": 5, "synapse_weight": 6.0},
            ]
        )
        dan_edges = pd.DataFrame(
            [
                {"source_id": 6, "target_id": 3, "synapse_weight": 3.0},
                {"source_id": 6, "target_id": 5, "synapse_weight": 2.0},
            ]
        )
        artifacts = self._write_outputs(
            nodes=nodes,
            edges=edges,
            dan_edges=dan_edges,
            mv=self.requested_mv or DEFAULT_MATERIALIZATION_VERSION,
            tables=TableDiscovery(
                synapse_table="mock_synapses",
                cell_tables=["mock_cells"],
            ),
        )
        return artifacts


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _extract_column_names(metadata: Mapping[str, Any]) -> List[str]:
    candidates: List[str] = []
    if "schema" in metadata and isinstance(metadata["schema"], Mapping):
        schema = metadata["schema"]
        if "properties" in schema and isinstance(schema["properties"], Mapping):
            candidates.extend(str(name).lower() for name in schema["properties"].keys())
        if "columns" in schema and isinstance(schema["columns"], list):
            for column in schema["columns"]:
                if isinstance(column, Mapping) and "name" in column:
                    candidates.append(str(column["name"]).lower())
                elif isinstance(column, str):
                    candidates.append(column.lower())
    if "columns" in metadata and isinstance(metadata["columns"], list):
        for column in metadata["columns"]:
            if isinstance(column, Mapping) and "name" in column:
                candidates.append(str(column["name"]).lower())
            elif isinstance(column, str):
                candidates.append(column.lower())
    if not candidates and "dtypes" in metadata and isinstance(metadata["dtypes"], Mapping):
        candidates.extend(str(name).lower() for name in metadata["dtypes"].keys())
    return sorted(set(candidates))


def _infer_cell_type(fields: Mapping[str, Any]) -> Optional[str]:
    joined = " ".join(str(value).lower() for value in fields.values() if isinstance(value, str))
    if not joined:
        return None
    if any(token in joined for token in ["kenyon", "kc", "kcn"]):
        return "KC"
    if any(token in joined for token in ["mbon", "mbon_"]):
        return "MBON"
    if any(token in joined for token in ["pam", "ppl1", "dan", "dopamin"]):
        return "DAN"
    if any(token in joined for token in ["pn", "projection", "glomerulus", "olfactory"]):
        return "PN"
    return None


def _resolve_weight_column(columns: Iterable[str]) -> str:
    prioritized = [
        "synapse_count",
        "syn_count",
        "size",
        "weight",
        "n_syn",
        "count",
    ]
    lower_cols = {c.lower(): c for c in columns}
    for candidate in prioritized:
        if candidate in lower_cols:
            return lower_cols[candidate]
    raise PipelineError("Could not infer a synapse weight column from table columns.")


def _position_columns(columns: Iterable[str]) -> List[str]:
    names = []
    for column in columns:
        col_lower = column.lower()
        if "position" in col_lower and column not in names:
            names.append(column)
    return names


def _normalize_positions(value: Any) -> List[Tuple[float, float, float]]:
    coords: List[Tuple[float, float, float]] = []
    if value is None:
        return coords
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3 and not isinstance(value[0], (list, tuple, np.ndarray)):
        return [(float(value[0]), float(value[1]), float(value[2]))]
    if isinstance(value, (list, tuple)):
        for element in value:
            coords.extend(_normalize_positions(element))
    return coords


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_arg_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(description="PGCN connectome cache builder")
    parser.add_argument("--datastack", default=DEFAULT_DATASTACK, help="FlyWire datastack name")
    parser.add_argument("--mv", type=int, default=DEFAULT_MATERIALIZATION_VERSION, help="Materialization version to request")
    parser.add_argument("--out", type=Path, default=Path("data") / "cache", help="Output directory for cache artifacts")
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN_PATH, help="Path to CAVE secret JSON")
    parser.add_argument("--use-sample-data", action="store_true", help="Generate deterministic sample cache instead of querying CAVE")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    pipeline = ConnectomePipeline(
        datastack=args.datastack,
        materialization_version=args.mv,
        cache_dir=args.out,
        token_path=args.token,
    )
    pipeline.run(use_sample_data=args.use_sample_data)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
