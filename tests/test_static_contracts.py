"""Ensure the type checker rejects invalid public LLM usage, not just accepts examples."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_invalid_llm_contracts_are_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/typecheck/negative_cases.py"
    expected = {(number, line.split("expected-error: ", 1)[1].strip()) for number, line in enumerate(fixture.read_text().splitlines(), start=1) if "# expected-error: " in line}
    config = tmp_path / "pyrightconfig.json"
    config.write_text(
        json.dumps(
            {
                "include": [os.path.relpath(fixture, tmp_path)],
                "extraPaths": [str(root)],
                "pythonVersion": "3.11",
                "typeCheckingMode": "standard",
            }
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "pyright", "--pythonpath", sys.executable, "--outputjson", "--project", str(config)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    diagnostics = json.loads(result.stdout)["generalDiagnostics"]
    actual = {(entry["range"]["start"]["line"] + 1, entry["rule"]) for entry in diagnostics}
    assert actual == expected, result.stdout
    assert all(Path(entry["file"]) == fixture for entry in diagnostics)
