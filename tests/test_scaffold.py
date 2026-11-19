"""Scaffold check: the package imports and exposes a version.

This test proves only that the package layout and pytest work. It does not
test any modeling code, because none exists yet.
"""

import ml_model_audit


def test_package_imports_with_version():
    assert isinstance(ml_model_audit.__version__, str)
    assert ml_model_audit.__version__.count(".") == 2
