"""Shared helper for the interactive hover cursors on plot canvases."""

from __future__ import annotations

from typing import Any


class RedrawOnChangeMixin:
    """Repaint only when the tooltip state actually changed.

    ``motion_notify_event`` fires on every mouse move (roughly 60-120/s), and
    calling ``canvas.draw_idle()`` unconditionally means a full matplotlib
    redraw per event. That is affordable on an idle process, but while a
    background worker is running a tight Python loop the GIL is contended and
    each redraw takes roughly an order of magnitude longer -- the repaints then
    queue up faster than they retire and the window stops responding. Moving the
    pointer within one bar/cell does not change what the tooltip says, so most
    of those redraws were redundant anyway.

    Subclasses must expose a ``canvas`` attribute and call
    :meth:`_redraw_if_changed` instead of ``canvas.draw_idle()``.
    """

    _last_redraw_key: Any = "<unset>"

    def _redraw_if_changed(self, key: Any) -> None:
        """Redraw only if ``key`` differs from the last painted state.

        Args:
            key: Hashable-ish summary of what the tooltip currently shows
                (``None`` means "hidden"). Compared with ``!=``.
        """
        if key == self._last_redraw_key:
            return
        self._last_redraw_key = key
        try:
            self.canvas.draw_idle()
        except Exception:  # noqa: BLE001 -- a dead canvas must not break hovering
            pass
