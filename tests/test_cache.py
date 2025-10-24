"""Tests for the connectome cache scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from requests import Response

from pgcn.connectome_pipeline import CacheArtifacts, ConnectomePipeline
from pgcn.connectome_pipeline import PipelineError


@pytest.fixture()
def sample_cache(tmp_path: Path) -> CacheArtifacts:
    pipeline = ConnectomePipeline(cache_dir=tmp_path)
    artifacts = pipeline.run(use_sample_data=True)
    return artifacts


def test_schema_presence(sample_cache: CacheArtifacts) -> None:
    nodes = pd.read_parquet(sample_cache.nodes)
    edges = pd.read_parquet(sample_cache.edges)
    dan_edges = pd.read_parquet(sample_cache.dan_edges)
    meta = json.loads(sample_cache.meta.read_text())

    assert set(["node_id", "type", "x", "y", "z"]).issubset(nodes.columns)
    assert set(["source_id", "target_id", "synapse_weight"]).issubset(edges.columns)
    assert set(["source_id", "target_id", "synapse_weight"]).issubset(dan_edges.columns)
    assert {
        "datastack",
        "materialization_version",
        "synapse_table",
        "cell_tables",
    }.issubset(meta.keys())


def test_graph_non_empty(sample_cache: CacheArtifacts) -> None:
    nodes = pd.read_parquet(sample_cache.nodes)
    edges = pd.read_parquet(sample_cache.edges)
    dan_edges = pd.read_parquet(sample_cache.dan_edges)

    assert not nodes.empty
    assert not edges.empty
    assert not dan_edges.empty
    assert (edges["synapse_weight"] > 0).all()
    assert (dan_edges["synapse_weight"] > 0).all()


def test_no_direct_pn_to_mbon_edges(sample_cache: CacheArtifacts) -> None:
    nodes = pd.read_parquet(sample_cache.nodes)
    edges = pd.read_parquet(sample_cache.edges)

    type_lookup = nodes.set_index("node_id")["type"].to_dict()
    pn_mbon_edges = [
        (src, tgt)
        for src, tgt in edges[["source_id", "target_id"]].itertuples(index=False)
        if type_lookup.get(int(src)) == "PN" and type_lookup.get(int(tgt)) == "MBON"
    ]
    assert not pn_mbon_edges, "Core subgraph must exclude direct PN→MBON edges"


def test_permission_error_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = ConnectomePipeline()
    monkeypatch.setattr(pipeline, "_read_token", lambda: "dummy-token")

    class StubClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            response = Response()
            response.status_code = 403
            raise requests.HTTPError("403 Client Error", response=response)

    monkeypatch.setattr("pgcn.connectome_pipeline.CAVEclient", StubClient)

    with pytest.raises(PipelineError) as excinfo:
        pipeline._init_client()

    assert "lacks 'view' permission" in str(excinfo.value)
