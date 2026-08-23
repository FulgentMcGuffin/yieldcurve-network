"""The one exception the analysis layer must never treat as a data error.

Long computations here take a ``progress`` callback and call it at checkpoints.
The GUI's callback raises :class:`ComputationCancelled` from inside it to unwind
a cancelled run, which means every ``except Exception`` guarding a per-item
failure sits directly in that exception's path -- and would silently turn a
cancellation into "skip this item and carry on".

Analysis code therefore re-raises this type explicitly before its own
``except Exception`` handlers. It lives here, rather than in the GUI's worker
module, so the analysis layer can name it without importing from the GUI.
"""

from __future__ import annotations


class ComputationCancelled(Exception):
    """Raised from a progress callback to abort a long-running computation."""
