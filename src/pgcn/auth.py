"""Helpers for provisioning FlyWire authentication tokens.

This module exposes a small CLI (``pgcn-auth``) that writes the FlyWire
CAVE token JSON expected by :mod:`pgcn.connectome_pipeline`.  It removes the
manual steps of creating ``~/.cloudvolume/secrets`` and hand-authoring the
``cave-secret.json`` payload, which have proven error prone during setup.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .connectome_pipeline import DEFAULT_TOKEN_PATH


class TokenProvisionError(RuntimeError):
    """Raised when the token CLI cannot resolve a valid token string."""


def write_token(token: str, path: Path, *, force: bool = False) -> Path:
    """Persist a FlyWire token JSON to ``path``.

    Parameters
    ----------
    token:
        FlyWire CAVE token string.  Leading/trailing whitespace is stripped
        prior to writing the secret.
    path:
        Filesystem location where ``{"token": "..."}`` should be written.
    force:
        When ``True`` the file is overwritten if it already exists.  The
        default behaviour is to refuse overwriting existing secrets to reduce
        accidental credential clobbering.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    ValueError
        If ``token`` is empty after trimming whitespace.
    FileExistsError
        When ``path`` already exists and ``force`` is ``False``.
    """

    token = token.strip()
    if not token:
        raise ValueError("Token string is empty; nothing to write.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        raise FileExistsError(
            f"Token file {path} already exists. Pass --force to overwrite it."
        )

    payload = {"token": token}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _load_token_from_file(path: Path) -> str:
    token = Path(path).read_text().strip()
    if not token:
        raise TokenProvisionError(f"Token file {path} is empty.")
    return token


def build_arg_parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        description="Write a FlyWire CAVE token to the expected cache location."
    )
    parser.add_argument(
        "--token",
        help=(
            "Token string. If omitted the command prompts securely. Use '-' to "
            "read from stdin."
        ),
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Path to a file containing the token string (whitespace ignored).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help="Destination JSON path for the secret (defaults to FlyWire's standard).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing secret at --path.",
    )
    return parser


def _resolve_token(args: argparse.Namespace) -> str:
    candidates = []
    if args.token_file is not None:
        candidates.append(_load_token_from_file(args.token_file))

    if args.token is not None:
        if args.token == "-":
            token = sys.stdin.read()
        else:
            token = args.token
        candidates.append(token)

    if candidates:
        return candidates[-1]

    # Prompt interactively as a last resort to avoid echoing secrets.
    token = getpass.getpass("FlyWire token: ")
    if not token.strip():
        raise TokenProvisionError("No token supplied via flag, file, stdin, or prompt.")
    return token


def cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    token = _resolve_token(args)
    try:
        path = write_token(token, args.path, force=args.force)
    except ValueError as exc:
        raise TokenProvisionError(str(exc)) from exc

    parser.exit(status=0, message=f"Token written to {path}\n")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    cli()
