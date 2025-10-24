from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from pgcn.flywire_access import FlywireAccessStatus, diagnose_flywire_access


def test_diagnose_success(monkeypatch, tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "abc"}')

    class DummyClient:  # pragma: no cover - trivial container
        def __init__(self, datastack_name: str, auth_token: str):
            assert datastack_name == "flywire_fafb_production"
            assert auth_token == "abc"
            self.info = SimpleNamespace(
                get_datastack_info=lambda datastack_name: {"dataset": "fafb"}
            )
            self.materialize = SimpleNamespace(get_versions=lambda: [1, 2, 3])

    monkeypatch.setattr("pgcn.flywire_access.CAVEclient", DummyClient)

    status = diagnose_flywire_access(extra_token_paths=[token_path])
    assert status.success
    assert status.token_source == f"file:{token_path}"
    assert status.dataset == "fafb"
    assert status.versions_ok
    assert status.versions_count == 3


def test_diagnose_http_403(monkeypatch, tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "abc"}')

    response = requests.Response()
    response.status_code = 403
    response._content = b'{"error": "missing_permission"}'

    def raising_client(*_, **__):
        raise requests.HTTPError(response=response)

    monkeypatch.setattr("pgcn.flywire_access.CAVEclient", raising_client)

    status = diagnose_flywire_access(extra_token_paths=[token_path])
    assert not status.success
    assert status.info_error is not None
    assert "HTTP 403" in status.info_error


@pytest.mark.parametrize(
    "status, expected",
    [
        (
            FlywireAccessStatus(
                datastack="flywire_fafb_production",
                token_source="file:/tmp/token.json",
                token_error="missing",
                info_ok=False,
                info_error="FlyWire token unavailable.",
                dataset=None,
                versions_ok=False,
                versions_error=None,
                versions_count=None,
            ),
            "FlyWire token unavailable",
        ),
        (
            FlywireAccessStatus(
                datastack="flywire_fafb_production",
                token_source="file:/tmp/token.json",
                token_error=None,
                info_ok=False,
                info_error="HTTP 403: missing permission",
                dataset=None,
                versions_ok=False,
                versions_error=None,
                versions_count=None,
            ),
            "lacks 'view' permission",
        ),
        (
            FlywireAccessStatus(
                datastack="flywire_fafb_production",
                token_source="file:/tmp/token.json",
                token_error=None,
                info_ok=False,
                info_error="HTTP 401: unauthorised",
                dataset=None,
                versions_ok=False,
                versions_error=None,
                versions_count=None,
            ),
            "HTTP 401",
        ),
    ],
)
def test_preflight_error_paths(monkeypatch, status, expected):
    from pgcn import connectome_pipeline

    pipeline = connectome_pipeline.ConnectomePipeline()

    def fake_diagnose(*_, **__):
        return status

    monkeypatch.setattr("pgcn.connectome_pipeline.diagnose_flywire_access", fake_diagnose)

    with pytest.raises(connectome_pipeline.PipelineError) as exc:
        pipeline._preflight_access_check()

    assert expected in str(exc.value)


def test_preflight_success(monkeypatch):
    from pgcn import connectome_pipeline

    status = FlywireAccessStatus(
        datastack="flywire_fafb_production",
        token_source="file:/tmp/token.json",
        token_error=None,
        info_ok=True,
        info_error=None,
        dataset="fafb",
        versions_ok=True,
        versions_error=None,
        versions_count=5,
    )

    pipeline = connectome_pipeline.ConnectomePipeline()

    monkeypatch.setattr(
        "pgcn.connectome_pipeline.diagnose_flywire_access",
        lambda *_, **__: status,
    )

    # Should not raise.
    pipeline._preflight_access_check()
