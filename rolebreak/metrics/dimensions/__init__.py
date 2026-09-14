"""Built-in RoleVoiceBench metrics — one module per dimension.

Each module defines a :class:`~rolebreak.metrics.base.Metric` and self-registers
via :func:`~rolebreak.metrics.registry.register_metric`. Importing this package
imports them all, so the registry is fully populated.
"""

from __future__ import annotations

import importlib
import pkgutil


def _load_all() -> None:
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")


_load_all()
