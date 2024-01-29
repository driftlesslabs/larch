from __future__ import annotations

_uidn = 0


def uid():
    global _uidn
    _uidn += 1
    return f"rx{_uidn}"
