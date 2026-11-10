"""Type aliases shared by the modules."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

type Pattern = str
type Term = tuple[Pattern, ...]
type Shape = tuple[int, ...]
type Marked = Sequence[int]
type Point = list[int]
type AdjacencyMatrix = list[list[int]]
type Equation = tuple[tuple[int, ...], int]
type IntArray = NDArray[np.int64]
type UIntArray = NDArray[np.uint64]
