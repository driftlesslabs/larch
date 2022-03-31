import numpy as np
from rich.table import Table
from rich.live import Live

from IPython.display import display, HTML

class Dashboard:

    def __init__(self, status="", params=None):
        self.title = "[bold blue]Larch Model Dashboard"
        self.status = status
        self.loglike = "Log Likelihood = [red italic]pending"
        if params is None:
            params = "[red]No parameters"
        self.params = params
        self._live = None
        self.tag = display(self.render(), display_id=True)

    def set_title(self, x):
        if x is not None:
            self.title = f"[bold blue]{x}"

    def set_loglike(self, x, best=None):
        if x is not None:
            if not np.isfinite(x):
                self.loglike = f"Log Likelihood = [red]{x:20.6f}"
            else:
                self.loglike = f"Log Likelihood = {x:20.6f}"
        if best is not None:
            self.loglike += f" [green]Best = {best:.6f}"

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

    def update_content(self, title=None, loglike=None, params=None, status=None, bestloglike=None):
        self.set_title(title)
        self.set_loglike(loglike, bestloglike)
        if params is not None:
            self.params = params
        if status is not None:
            self.status = status
        self.tag.update(self.render())

    def clear(self):
        self.tag.update(HTML(""))
