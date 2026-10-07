import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVOCATION_MANIFEST = {
    "src/array_processing_re.py": "operational",
    "src/utils/inspect_laz.py": "diagnostic",
}


def test_invocation_manifest_covers_every_main_guard():
    guarded_files = set()
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in ast.walk(tree)
        ):
            guarded_files.add(path.relative_to(PROJECT_ROOT).as_posix())

    assert guarded_files == set(INVOCATION_MANIFEST)


@pytest.mark.parametrize(
    "entry",
    [
        ["src/array_processing_re.py"],
        ["-m", "src.array_processing_re"],
        ["src/utils/inspect_laz.py"],
        ["-m", "src.utils.inspect_laz"],
    ],
)
def test_tool_help_works_without_side_effects(entry, tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "--input-path" in result.stdout
    assert not list(tmp_path.iterdir())
