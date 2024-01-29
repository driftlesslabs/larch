from __future__ import annotations

import time

import numpy as np
from IPython.display import HTML, display
from rich.live import Live
from rich.table import Table


class Dashboard:
    def __init__(self, status="", params=None, show=True, throttle=0.1):
        self.title = "[bold blue]Larch Model Dashboard"
        self.status = status
        self.loglike = "Log Likelihood = [red italic]pending"
        if params is None:
            params = "[red]No parameters"
        self.params = params
        self._live = None
        self.display(show)
        self.tick = time.time()
        self._throttle = throttle

    def display(self, show=True):
        if show:
            self.tag = display(self.render(), display_id=True)
        else:
            self.clear()
            self.tag = None

    def set_title(self, x):
        if x is not None:
            self.title = f"[bold blue]{x}"

    def set_loglike(self, x, best=None):
        if x is not None:
            if not np.isfinite(x):
                if best is not None:
                    self.loglike = f"Log Likelihood [red]Current = {x:17.6f}"
                else:
                    self.loglike = f"Log Likelihood = [red]{x:24.6f}"
            else:
                if best is not None:
                    self.loglike = f"Log Likelihood [yellow]Current = {x:17.6f}"
                else:
                    self.loglike = f"Log Likelihood = {x:24.6f}"
        if best is not None:
            self.loglike += f" [green]Best = {best:17.6f}"

    def render(self):
        t = Table()
        t.add_column("[bold blue]Larch Model Dashboard")
        if self.status:
            t.add_row(self.status)
        t.add_row(self.loglike)
        t.add_row(self.params)
        if self._live is not None:
            self._live.update(t)
        return t

    def __enter__(self):
        self._live = Live(self.render())
        self._live.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._live.__exit__(exc_type, exc_val, exc_tb)

    def update_content(
        self, title=None, loglike=None, params=None, status=None, bestloglike=None
    ):
        self.set_title(title)
        self.set_loglike(loglike, bestloglike)
        if params is not None:
            self.params = params
        if status is not None:
            self.status = status
        if self.tag is not None:
            self.tag.update(self.render())

    def clear(self):
        try:
            tag = self.tag
        except AttributeError:
            tag = None
        if tag is not None:
            self.tag.update(HTML(""))

    def throttle(self):
        now = time.time()
        if now < self.tick + self._throttle:
            return False
        else:
            self.tick = now
            return True
