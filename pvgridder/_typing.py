from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias


VectorLike: TypeAlias = Union[
    Sequence[Union[int, float]],
    NDArray[Union[np.integer[Any], np.floating[Any]]],
]
MatrixLike: TypeAlias = Union[
    Sequence[Sequence[Union[int, float]]],
    np.ndarray[
        Tuple[int, int],
        np.dtype[Union[np.integer[Any], np.floating[Any]]],
    ],
]
