"""Integer arithmetic coding and greedy rendering of a prescribed byte prefix."""

from bisect import bisect_right
from collections.abc import Iterator, Sequence

from numpy import int32
from numpy.typing import NDArray

type _CDF = Sequence[int] | NDArray[int32]

_BITS = 32
_FULL = 1 << _BITS
_HALF = _FULL // 2
_QUARTER = _HALF // 2
MAX_TOTAL = _QUARTER


class _Interval:
    """Inclusive bounds on the coder's 32-bit grid."""

    def __init__(self, low: int = 0, high: int = _FULL - 1) -> None:
        self.low = low
        self.high = high

    def narrow(self, cdf: _CDF, symbol: int) -> None:
        self.low, self.high = self.bounds(cdf, symbol)

    def bounds(self, cdf: _CDF, symbol: int) -> tuple[int, int]:
        width = self.high - self.low + 1
        total = int(cdf[-1])
        return (
            self.low + width * int(cdf[symbol]) // total,
            self.low + width * int(cdf[symbol + 1]) // total - 1,
        )

    def symbol_at(self, cdf: _CDF, point: int) -> int:
        width = self.high - self.low + 1
        # Invert bounds()'s floor division at an exact symbol boundary.
        offset = point - self.low
        value = ((offset + 1) * int(cdf[-1]) - 1) // width
        return bisect_right(cdf, value) - 1

    def renormalize(self) -> Iterator[int]:
        """Shift into the full grid, yielding the origin of each shift."""
        while True:
            if self.high < _HALF:
                offset = 0
            elif self.low >= _HALF:
                offset = _HALF
            elif self.low >= _QUARTER and self.high < 3 * _QUARTER:
                offset = _QUARTER  # Straddles the midpoint: defer the bit.
            else:
                return
            self.low = 2 * (self.low - offset)
            self.high = 2 * (self.high - offset) + 1
            yield offset


class ArithmeticEncoder:
    """Emit settled bits; use finish() only when a final code point is needed.

    Frequency tables start at zero, increase strictly, and end at most at MAX_TOTAL.
    """

    def __init__(self) -> None:
        self._interval = _Interval()
        self._deferred_bits = 0
        self._data = bytearray()
        self._byte = 0
        self._used = 0

    def _write_bit(self, bit: int) -> None:
        self._byte = (self._byte << 1) | bit
        self._used += 1
        if self._used == 8:
            self._data.append(self._byte)
            self._byte = self._used = 0

    def _emit(self, bit: int) -> None:
        self._write_bit(bit)
        # After an underflow shift, the resolved bit is followed by its complement.
        for _ in range(self._deferred_bits):
            self._write_bit(1 - bit)
        self._deferred_bits = 0

    def write(self, cdf: _CDF, symbol: int) -> None:
        self._interval.narrow(cdf, symbol)
        for offset in self._interval.renormalize():
            if offset == _QUARTER:
                self._deferred_bits += 1
            else:
                self._emit(int(offset == _HALF))

    def settled_bytes(self) -> bytes:
        """Return the whole bytes fixed by every point in the current interval."""
        return bytes(self._data)

    def finish(self) -> bytes:
        """Choose a point in the final interval and pad to a whole byte."""
        self._deferred_bits += 1
        self._emit(int(self._interval.low >= _QUARTER))
        if self._used:
            self._data.append(self._byte << (8 - self._used))
        return bytes(self._data)


class ArithmeticDecoder:
    """Read symbols from one code point, treating missing tail bits as zeros."""

    def __init__(self, data: bytes | memoryview) -> None:
        self._interval = _Interval()
        self._data = data
        self._position = 0
        self._code = 0
        for _ in range(_BITS):
            self._code = (self._code << 1) | self._read_bit()

    def _read_bit(self) -> int:
        byte, bit = divmod(self._position, 8)
        if byte >= len(self._data):
            return 0  # The finite binary fraction has an infinite zero tail.
        self._position += 1
        return (self._data[byte] >> (7 - bit)) & 1

    def read(self, cdf: _CDF) -> int:
        symbol = self._interval.symbol_at(cdf, self._code)
        self._interval.narrow(cdf, symbol)
        for shift in self._interval.renormalize():
            self._code = 2 * (self._code - shift) + self._read_bit()
        return symbol


class PrefixInterval:
    """Track all code points still compatible with the requested byte prefix.

    The coder interval uses an inclusive integer grid. The prefix overlap uses
    half-open bounds with ``scale`` extra bits so long prefixes stay exact.
    """

    def __init__(
        self, interval: _Interval, low: int, high: int, scale: int
    ) -> None:
        self._interval = interval
        self._low = low
        self._high = high  # Exclusive; endpoints use units of 2**-scale.
        self._scale = scale

    @classmethod
    def from_bytes(cls, data: bytes) -> "PrefixInterval":
        prefix_bits = 8 * len(data)
        scale = max(0, prefix_bits - _BITS)
        prefix_width = 1 << max(0, _BITS - prefix_bits)
        start = int.from_bytes(data, "big") * prefix_width
        return cls(_Interval(), start, start + prefix_width, scale)

    @property
    def contained(self) -> bool:
        """Whether every code point for the chosen symbols has the target prefix."""
        return (
            self._low == self._interval.low << self._scale
            and self._high == (self._interval.high + 1) << self._scale
        )

    def read(self, cdf: _CDF) -> int:
        """Select the symbol with the largest overlap and advance in place."""
        interval, scale = self._interval, self._scale
        first_cell = self._low >> scale
        last_cell = (self._high - 1) >> scale
        first_symbol = interval.symbol_at(cdf, first_cell)
        last_symbol = first_symbol if first_cell == last_cell else interval.symbol_at(cdf, last_cell)

        # Only the first and last candidates may be clipped by the target bounds.
        symbol = first_symbol
        symbol_low, symbol_high = interval.bounds(cdf, symbol)
        overlap_start = self._low
        overlap_end = min(self._high, (symbol_high + 1) << scale)
        for candidate in range(first_symbol + 1, last_symbol + 1):
            candidate_low, candidate_high = interval.bounds(cdf, candidate)
            start = candidate_low << scale
            end = min(self._high, (candidate_high + 1) << scale)
            # Keep the earlier symbol on ties so the emitted cover is stable.
            if end - start > overlap_end - overlap_start:
                symbol = candidate
                symbol_low, symbol_high = candidate_low, candidate_high
                overlap_start, overlap_end = start, end

        interval.low, interval.high = symbol_low, symbol_high
        for offset in interval.renormalize():
            # Apply the same grid change to the finer-grained overlap.
            scaled_offset = offset << scale
            overlap_start = 2 * (overlap_start - scaled_offset)
            overlap_end = 2 * (overlap_end - scaled_offset)
        self._low, self._high = overlap_start, overlap_end
        return symbol
