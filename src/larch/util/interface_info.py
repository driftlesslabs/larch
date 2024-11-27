from __future__ import annotations

import os
import sys

from .. import __version__


class Info:
    def __init__(
        self, appname="Larch", extra=True, version=None, path=None, minimal=False
    ):
        self.appname = appname
        self.extra = extra
        self.version = version or __version__
        self.minimal = minimal
        from .. import __path__

        self.path = path or __path__[0]

    def __repr__(self):
        r = f"┌── {self.appname.upper()} {self.version} " + "─" * (
            57 - len(self.version)
        )
        v = "\n│".join(sys.version.split("\n"))
        r += f"\n│Python {v}"
        r += f"\n│EXE ─ {sys.executable}"
        r += f"\n│CWD ─ {os.getcwd()}"
        for p in sys.path[:1]:
            r += f"\n│PTH ┬ {p}"
        for p in sys.path[1:-1]:
            r += f"\n│    ├ {p}"
        for p in sys.path[-1:]:
            r += f"\n│    └ {p}"
        r += "\n└───────────────────────────────────────────────────────────────────────────"
        return r

    def _repr_html_(self):
        from ..util.styles import _default_css_jupyter

        style_prefix = f"<style>{_default_css_jupyter}</style>\n"
        from xmle import Elem

        xsign = Elem("div", {"class": "larch_head_tag"})
        from .images import favicon

        p = xsign.elem("p", {"style": "margin-top:6px"})
        p.elem(
            "img",
            {
                "width": "32",
                "height": "32",
                "src": f"data:image/png;base64,{favicon}",
                "style": "float:left;position:relative;top:-3px;padding-right:0.2em;",
            },
            tail=f" {self.appname} ",
        )
        p.elem("span", {"class": "larch_head_tag_ver"}, text=f" {self.version} ")
        if not self.minimal:
            p.elem("span", {"class": "larch_head_tag_pth"}, text=f" {self.path} ")

            if self.extra:
                v = "\n│".join(sys.version.split("\n"))
                # xsign.elem("br")
                xinfo = xsign.elem(
                    "div",
                    {
                        "class": "larch_head_tag_more",
                        "style": "padding:7px",
                    },
                    text=f"Python {v}",
                )
                xinfo.elem("br", tail=f"EXE - {sys.executable}")
                xinfo.elem("br", tail=f"CWD - {os.getcwd()}")
                xinfo.elem("br", tail="PATH - ")
                ul = xinfo.elem("ul", {"style": "margin-top:0; margin-bottom:0;"})
                for p in sys.path:
                    ul.elem("li", text=p)

        return style_prefix + xsign.tostring()

    def __call__(self, *args, **kwargs):
        """
        Do nothing (for now).

        Returns
        -------
        self
        """
        return self


def ipython_status(magic_matplotlib=False):
    message_set = set()
    try:
        # This will work in iPython, and fail otherwise
        cfg = get_ipython().config  # noqa: F821
    except Exception:
        message_set.add("Not IPython")
    else:
        import IPython
        import IPython.core.error

        message_set.add("IPython")

        if magic_matplotlib:
            try:
                get_ipython().magic("matplotlib inline")  # noqa: F821
            except (IPython.core.error.UsageError, KeyError):
                message_set.add("IPython inline plotting not available")

        # Caution: cfg is an IPython.config.loader.Config
        if cfg["IPKernelApp"]:
            message_set.add("IPython QtConsole")
            try:
                if cfg["IPKernelApp"]["pylab"] == "inline":
                    message_set.add("pylab inline")
                else:
                    message_set.add("pylab loaded but not inline")
            except Exception:
                message_set.add("pylab not loaded")
        elif cfg["TerminalIPythonApp"]:
            try:
                if cfg["TerminalIPythonApp"]["pylab"] == "inline":
                    message_set.add("pylab inline")
                else:
                    message_set.add("pylab loaded but not inline")
            except Exception:
                message_set.add("pylab not loaded")
    return message_set
