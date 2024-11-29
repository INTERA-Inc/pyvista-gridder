from __future__ import annotations
from typing import Callable, Optional

import importlib
from functools import wraps


def require_package(name: str, version: Optional[tuple[int]] = None) -> Callable:
    """Decorator to check if required package is installed."""
    def decorator(func: Callable):
        """Decorate function."""
        try:
            importlib.import_module(name)

            if version:
                package_version = importlib.metadata.version(name)
                package_version = tuple(map(int, package_version.split(".")))

                if package_version < version:
                    raise ModuleNotFoundError()

        except ModuleNotFoundError:
            name_version = (
                f"{name}>={'.'.join(map(str, version))}"
                if version
                else name
            )

            raise ModuleNotFoundError(f"{func.__name__} requires package '{name_version}'")

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrap function."""
            return func(*args, **kwargs)
            
        return wrapper

    return decorator
