from __future__ import annotations

import base64
import io

from PIL import Image
from xmle import Elem


def make_png(
    content,
    dpi=None,
    compress=True,
    output="Elem",
    close_after=True,
    pad_inches=0.1,
    facecolor=None,
    return_size=False,
    return_dpi=False,
):
    _bytes_io = None
    _img = None
    _size = None

    if hasattr(content, "_repr_png_"):
        content = content._repr_png_()

    if isinstance(content, bytes):
        _bytes_io = io.BytesIO(content)

    if "matplotlib" in str(type(content)):
        import matplotlib.figure

        from .._optional import pyplot as plt

        if not isinstance(content, matplotlib.figure.Figure):
            try:
                content = content.get_figure()
            except AttributeError:
                if not hasattr(content, "savefig"):
                    raise TypeError(
                        "matplotlib content must provide `get_figure` or `savefig` method."
                    ) from None

        # content is a Figure or otherwise has a savefig method
        try:
            fig_number = content.number
        except AttributeError:
            fig_number = None

        if facecolor is None:
            try:
                facecolor = content.get_facecolor()
            except Exception:
                facecolor = "w"

        try:
            edgecolor = content.get_edgecolor()
        except Exception:
            edgecolor = "none"

        _bytes_io = io.BytesIO()
        content.savefig(
            _bytes_io,
            dpi=dpi,
            orientation="portrait",
            format="png",
            bbox_inches="tight",
            pad_inches=pad_inches,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

        if close_after and fig_number is not None:
            plt.close(fig_number)

    if _bytes_io.getvalue()[:4] == b"\x89PNG":
        _img = Image.open(_bytes_io)
        dpi = _img.info.get("dpi", dpi)
        if dpi is None:
            dpi = [96, 96]
        if compress:
            _img = _img.convert(mode="P", palette="ADAPTIVE")
        _bytes_io = io.BytesIO()
        _img.save(_bytes_io, dpi=dpi, format="png")
        _size = _img.size
    else:
        raise ValueError("Not valid PNG data")

    if output.lower() == "elem":
        result = Elem(
            tag="img",
            src=f"data:image/png;base64,{base64.standard_b64encode(_bytes_io.getvalue()).decode()}",
        )
    elif output.lower() == "bytesio":
        result = _bytes_io
    else:
        result = _img

    if return_size:
        if isinstance(result, tuple):
            result = (*result, _size)
        else:
            result = result, _size

    if return_dpi:
        if isinstance(result, tuple):
            result = (*result, dpi)
        else:
            result = result, dpi

    return result
