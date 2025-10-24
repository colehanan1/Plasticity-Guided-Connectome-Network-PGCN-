"""FlyWire access diagnostics and CLI helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

try:  # pragma: no cover - optional dependency resolved at runtime
    from caveclient import CAVEclient  # type: ignore
except ImportError:  # pragma: no cover - exercised when caveclient missing
    CAVEclient = None  # type: ignore


DEFAULT_DATASTACK = "flywire_fafb_production"
PRIMARY_SECRET = Path.home() / ".cloudvolume/secrets/global.daf-apis.com-cave-secret.json"
LEGACY_SECRET = Path.home() / ".cloudvolume/secrets/cave-secret.json"
ENV_VARS = ("CHUNKEDGRAPH_SECRET", "CAVE_TOKEN", "FLYWIRE_TOKEN")


@dataclass(slots=True)
class FlywireAccessStatus:
    """Structured result produced by :func:`diagnose_flywire_access`."""

    datastack: str
    token_source: Optional[str]
    token_error: Optional[str]
    info_ok: bool
    info_error: Optional[str]
    dataset: Optional[str]
    versions_ok: bool
    versions_error: Optional[str]
    versions_count: Optional[int]

    @property
    def success(self) -> bool:
        """Return ``True`` when the datastack metadata call succeeded."""

        return self.info_ok


def _iter_candidate_paths(extra_paths: Iterable[Path | str] | None = None) -> Iterable[Path]:
    if extra_paths:
        for path in extra_paths:
            yield Path(path).expanduser()
    yield PRIMARY_SECRET
    yield LEGACY_SECRET


def _load_token_from_path(path: Path) -> Optional[str]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - exercised when JSON invalid
        raise ValueError(f"Failed to parse {path}: {exc}")
    token = payload.get("token")
    if not token:
        raise ValueError(f"No 'token' key present in {path}")
    return str(token)


def _discover_token(candidate_paths: Iterable[Path | str] | None = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Search known locations for a FlyWire token."""

    path_error: Optional[str] = None
    for path in _iter_candidate_paths(candidate_paths):
        try:
            token = _load_token_from_path(path)
        except ValueError as exc:
            path_error = str(exc)
            continue
        if token:
            return token, f"file:{path}", None

    for env in ENV_VARS:
        token = os.getenv(env)
        if token:
            return token, f"env:{env}", None

    return None, None, path_error or "Token not found in default secret locations."


def _format_http_error(exc: requests.HTTPError) -> str:
    response = exc.response
    status = response.status_code if response is not None else "<unknown>"
    message = response.text.strip() if response is not None else str(exc)
    return f"HTTP {status}: {message}"


def diagnose_flywire_access(
    datastack: str = DEFAULT_DATASTACK,
    *,
    extra_token_paths: Iterable[Path | str] | None = None,
) -> FlywireAccessStatus:
    """Inspect token availability and FlyWire permissions for ``datastack``."""

    token, token_source, token_error = _discover_token(extra_token_paths)

    if CAVEclient is None:  # pragma: no cover - depends on optional install
        return FlywireAccessStatus(
            datastack=datastack,
            token_source=token_source,
            token_error=token_error or "caveclient is not installed.",
            info_ok=False,
            info_error="caveclient is unavailable; install it to continue.",
            dataset=None,
            versions_ok=False,
            versions_error="caveclient is unavailable; install it to continue.",
            versions_count=None,
        )

    if not token:
        return FlywireAccessStatus(
            datastack=datastack,
            token_source=token_source,
            token_error=token_error,
            info_ok=False,
            info_error="FlyWire token unavailable.",
            dataset=None,
            versions_ok=False,
            versions_error="FlyWire token unavailable.",
            versions_count=None,
        )

    try:
        client = CAVEclient(datastack_name=datastack, auth_token=token)
    except requests.HTTPError as exc:  # pragma: no cover - depends on live service
        return FlywireAccessStatus(
            datastack=datastack,
            token_source=token_source,
            token_error=None,
            info_ok=False,
            info_error=_format_http_error(exc),
            dataset=None,
            versions_ok=False,
            versions_error=_format_http_error(exc),
            versions_count=None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return FlywireAccessStatus(
            datastack=datastack,
            token_source=token_source,
            token_error=None,
            info_ok=False,
            info_error=f"Failed to initialise CAVEclient: {exc}",
            dataset=None,
            versions_ok=False,
            versions_error=f"Failed to initialise CAVEclient: {exc}",
            versions_count=None,
        )

    try:
        info = client.info.get_datastack_info(datastack_name=datastack)
        dataset = info.get("dataset") if isinstance(info, dict) else None
    except requests.HTTPError as exc:  # pragma: no cover - depends on live service
        return FlywireAccessStatus(
            datastack=datastack,
            token_source=token_source,
            token_error=None,
            info_ok=False,
            info_error=_format_http_error(exc),
            dataset=None,
            versions_ok=False,
            versions_error=_format_http_error(exc),
            versions_count=None,
        )
    except Exception as exc:
        return FlywireAccessStatus(
            datastack=datastack,
            token_source=token_source,
            token_error=None,
            info_ok=False,
            info_error=f"Failed to query datastack info: {exc}",
            dataset=None,
            versions_ok=False,
            versions_error=f"Failed to query datastack info: {exc}",
            versions_count=None,
        )

    versions_ok = False
    versions_error: Optional[str] = None
    versions_count: Optional[int] = None

    try:
        versions = client.materialize.get_versions()
        versions_count = len(list(versions))
        versions_ok = True
    except Exception as exc:  # pragma: no cover - depends on live service
        versions_error = f"Failed to list materialization versions: {exc}"

    return FlywireAccessStatus(
        datastack=datastack,
        token_source=token_source,
        token_error=None,
        info_ok=True,
        info_error=None,
        dataset=dataset,
        versions_ok=versions_ok,
        versions_error=versions_error,
        versions_count=versions_count,
    )


def _format_status(status: FlywireAccessStatus) -> str:
    lines = [f"Datastack: {status.datastack}"]
    lines.append(f"Token source: {status.token_source or 'not found'}")
    if status.token_error:
        lines.append(f"Token error: {status.token_error}")
    if status.info_ok:
        lines.append(f"InfoService: OK (dataset={status.dataset or 'unknown'})")
    else:
        lines.append(f"InfoService: FAILED ({status.info_error})")
    if status.versions_ok:
        lines.append(
            "Materialization: OK" + (
                f" ({status.versions_count} versions discovered)"
                if status.versions_count is not None
                else ""
            )
        )
    else:
        lines.append(
            "Materialization: FAILED" + (
                f" ({status.versions_error})" if status.versions_error else ""
            )
        )
    return "\n".join(lines)


def cli() -> int:
    """Console entry point to diagnose FlyWire access issues."""

    import argparse

    parser = argparse.ArgumentParser(description="Diagnose FlyWire datastack access")
    parser.add_argument(
        "--datastack",
        default=DEFAULT_DATASTACK,
        help="Datastack name to probe (default: %(default)s)",
    )
    parser.add_argument(
        "--token-path",
        action="append",
        dest="token_paths",
        help="Additional token paths to consult before the defaults.",
    )
    args = parser.parse_args()

    status = diagnose_flywire_access(args.datastack, extra_token_paths=args.token_paths)
    print(_format_status(status))
    return 0 if status.success else 1


__all__ = [
    "DEFAULT_DATASTACK",
    "FlywireAccessStatus",
    "diagnose_flywire_access",
    "cli",
]

