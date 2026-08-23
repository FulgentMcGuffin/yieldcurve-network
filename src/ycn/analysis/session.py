"""Save and reload a whole analysis: the settings that produced it, and its data.

A session is a zip archive holding one ``manifest.json`` and one Parquet file
per DataFrame:

```text
manifest.json          format version, timestamp, settings, per-stage scalars
frames/<name>.parquet  one per rendered table
```

**Figures are deliberately not stored.** Every figure in this application is a
pure function of a frame plus the settings, so re-rendering on load is exact,
keeps archives small, and -- unlike pickling matplotlib objects -- does not
break when matplotlib is upgraded. It also means a session never carries
executable state: loading one runs no code from the file.

The manifest is plain JSON so an archive stays inspectable with a text editor
and readable by anything that can open a zip, not just this application.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

# Bumped when the archive layout changes incompatibly. Readers refuse a version
# they do not understand rather than mis-parsing it.
FORMAT_VERSION = 1

SUFFIX = ".ycn"
FILE_FILTER = f"YieldCurve-Network analysis (*{SUFFIX});;All files (*.*)"

_MANIFEST = "manifest.json"
_FRAME_DIR = "frames"


class SessionError(RuntimeError):
    """Raised when an archive cannot be written or is not readable."""


@dataclass
class Session:
    """One saved analysis.

    Args:
        settings: Everything needed to restore the sidebar and describe the run.
        scalars: Per-stage non-table values (layer names, counts, orderings),
            keyed by stage: ``"mln"``, ``"residual"``, ``"evolution"``.
        frames: Rendered tables, keyed by ``"<stage>.<name>"``.
        saved_at: ISO timestamp written at save time.
    """

    settings: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, dict[str, Any]] = field(default_factory=dict)
    frames: dict[str, pl.DataFrame] = field(default_factory=dict)
    saved_at: str = ""

    def stages(self) -> list[str]:
        """Stages this session actually carries results for."""
        return sorted(self.scalars)

    def frame(self, key: str) -> pl.DataFrame:
        """A stored frame, or an empty one when the stage did not produce it."""
        return self.frames.get(key, pl.DataFrame())


def _jsonable(value: Any) -> Any:
    """Convert values the GUI holds into JSON-safe equivalents."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value)]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "value") and hasattr(value, "name"):  # enum-like
        return value.value
    return value


def save_session(path: Path | str, session: Session) -> Path:
    """Write ``session`` to ``path`` as a zip archive.

    The archive is built in memory and written once, so an interrupted save
    cannot leave a half-written file where a valid one used to be.
    """
    target = Path(path)
    if target.suffix.lower() != SUFFIX:
        target = target.with_suffix(SUFFIX)

    manifest = {
        "format_version": FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "settings": _jsonable(session.settings),
        "scalars": _jsonable(session.scalars),
        "frames": sorted(session.frames),
    }

    buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST, json.dumps(manifest, indent=2))
            for name, frame in session.frames.items():
                sink = io.BytesIO()
                frame.write_parquet(sink)
                archive.writestr(f"{_FRAME_DIR}/{name}.parquet", sink.getvalue())
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user verbatim
        raise SessionError(f"Could not build the analysis archive: {exc}") from exc

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(buffer.getvalue())
    except OSError as exc:
        raise SessionError(f"Could not write {target}: {exc}") from exc
    return target


def load_session(path: Path | str) -> Session:
    """Read a session archive written by :func:`save_session`."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            try:
                manifest = json.loads(archive.read(_MANIFEST).decode("utf-8"))
            except KeyError as exc:
                raise SessionError(
                    f"{source.name} is not a YieldCurve-Network analysis "
                    "(no manifest.json inside)."
                ) from exc

            version = manifest.get("format_version")
            if version != FORMAT_VERSION:
                raise SessionError(
                    f"{source.name} was written in format version {version}; "
                    f"this build reads version {FORMAT_VERSION}."
                )

            frames: dict[str, pl.DataFrame] = {}
            for name in manifest.get("frames", []):
                entry = f"{_FRAME_DIR}/{name}.parquet"
                try:
                    payload = archive.read(entry)
                except KeyError:
                    # A frame named in the manifest but missing from the zip:
                    # treat as empty rather than failing the whole load, so a
                    # partially truncated archive still opens.
                    continue
                frames[name] = pl.read_parquet(io.BytesIO(payload))
    except SessionError:
        raise
    except zipfile.BadZipFile as exc:
        raise SessionError(f"{source.name} is not a readable archive.") from exc
    except OSError as exc:
        raise SessionError(f"Could not read {source}: {exc}") from exc

    return Session(
        settings=manifest.get("settings", {}),
        scalars=manifest.get("scalars", {}),
        frames=frames,
        saved_at=manifest.get("saved_at", ""),
    )


def describe(session: Session) -> str:
    """One-line summary of an archive, for the status bar and the log."""
    parts = []
    for stage in session.stages():
        rows = sum(
            frame.height
            for key, frame in session.frames.items()
            if key.startswith(f"{stage}.")
        )
        parts.append(f"{stage} ({rows:,} rows)")
    return ", ".join(parts) if parts else "no stored results"
