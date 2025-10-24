from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgcn import auth


def test_write_token_creates_parent_and_json(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "cave-secret.json"
    written = auth.write_token("  example-token  ", target)

    assert written == target
    data = json.loads(target.read_text())
    assert data == {"token": "example-token"}


def test_write_token_requires_force_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "secret.json"
    auth.write_token("token-a", target)

    with pytest.raises(FileExistsError):
        auth.write_token("token-b", target)

    auth.write_token("token-b", target, force=True)
    assert json.loads(target.read_text()) == {"token": "token-b"}


@pytest.mark.parametrize("use_flag", [True, False])
def test_cli_writes_token(tmp_path: Path, use_flag: bool, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "secret.json"
    token_file = tmp_path / "token.txt"
    token_file.write_text("cli-token\n")

    argv = ["--path", str(target)]
    if use_flag:
        argv.extend(["--token", "cli-token"])
    else:
        argv.extend(["--token-file", str(token_file)])

    with pytest.raises(SystemExit) as exc:
        auth.cli(argv)

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Token written" in captured.out or "Token written" in captured.err
    assert json.loads(target.read_text()) == {"token": "cli-token"}
