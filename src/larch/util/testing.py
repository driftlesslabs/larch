from __future__ import annotations

import re


def _mash(x):
    return re.sub(r"\s+", " ", str(x)).strip()


def assert_same_text(x, y):
    assert _mash(x) == _mash(y)
