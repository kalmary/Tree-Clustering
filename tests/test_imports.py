import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tree_segmenter_uses_own_ray_module_without_laspy():
    code = """
import builtins
import sys

original_import = builtins.__import__

def import_without_laspy(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'laspy', 'pyvista'}:
        raise ImportError(f'{name} is not available')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_laspy
from src.array_processing_re import TreeSegmRay, get_rays
assert TreeSegmRay.__module__ == 'src.array_processing_re'
assert get_rays.__module__ == 'src.utils.get_rays'
assert 'laspy' not in sys.modules
assert 'pyvista' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ray_dependency_import_error_is_not_hidden_by_fallback():
    code = """
import builtins
import importlib

original_import = builtins.__import__

def import_with_ray_error(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'utils.get_rays' and level == 1:
        raise ImportError('ray dependency failed')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_with_ray_error
try:
    importlib.import_module('src.array_processing_re')
except ImportError as error:
    assert str(error) == 'ray dependency failed', str(error)
else:
    raise AssertionError('ray dependency import unexpectedly succeeded')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_segmenter_imports_from_parent_project():
    code = """
from src.tree_clustering.src.array_processing_re import TreeSegmRay, get_rays
assert TreeSegmRay.__module__ == 'src.tree_clustering.src.array_processing_re'
assert get_rays.__module__ == 'src.tree_clustering.src.utils.get_rays'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT.parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
