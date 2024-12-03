from __future__ import annotations

import contextlib
import os
import sys


@contextlib.contextmanager
def shush(stdout=False, stderr=True):
    """Suppress stdout and/or stderr using a context manager."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if stdout:
            sys.stdout = devnull
        if stderr:
            sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
