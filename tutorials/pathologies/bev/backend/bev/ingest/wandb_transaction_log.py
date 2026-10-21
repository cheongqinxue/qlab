"""Read W&B ``run-*.wandb`` transaction logs (LevelDB-style blocks).

Adapted from Weights & Biases SDK ``datastore.py`` (v0.22.x), trimmed for
read-only scanning without wandb internal-process checks.
"""

from __future__ import annotations

import logging
import os
import struct
import zlib
from typing import IO, Any

logger = logging.getLogger(__name__)

LEVELDBLOG_HEADER_LEN = 7
LEVELDBLOG_BLOCK_LEN = 32768
LEVELDBLOG_DATA_LEN = LEVELDBLOG_BLOCK_LEN - LEVELDBLOG_HEADER_LEN

LEVELDBLOG_FULL = 1
LEVELDBLOG_FIRST = 2
LEVELDBLOG_MIDDLE = 3
LEVELDBLOG_LAST = 4

LEVELDBLOG_HEADER_IDENT = ":W&B"
LEVELDBLOG_HEADER_MAGIC = 0xBEE1
LEVELDBLOG_HEADER_VERSION = 0


def _strtobytes(x: str) -> bytes:
    return bytes(x, "iso8859-1")


class WandbTransactionLog:
    """Scan a ``.wandb`` file and yield serialized protobuf record payloads."""

    def __init__(self) -> None:
        self._opened_for_scan = False
        self._fp: IO[bytes] | None = None
        self._index = 0
        self._size_bytes = 0
        self._fname = ""

        self._crc = [0] * (LEVELDBLOG_LAST + 1)
        for dtype in range(1, LEVELDBLOG_LAST + 1):
            self._crc[dtype] = zlib.crc32(_strtobytes(chr(dtype))) & 0xFFFFFFFF

    @property
    def byte_offset(self) -> int:
        """Current read offset (for incremental tail bookmarks)."""
        return self._index

    def open_for_scan(self, fname: str) -> None:
        """Open ``fname`` for read-only scanning from the start."""
        self._fname = fname
        self._fp = open(fname, "r+b")
        self._index = 0
        self._size_bytes = os.stat(fname).st_size
        self._opened_for_scan = True
        self._read_header()

    def seek(self, offset: int) -> None:
        """Seek to ``offset`` bytes from the file start."""
        if self._fp is None:
            msg = "file not open"
            raise RuntimeError(msg)
        self._fp.seek(offset)
        self._index = offset

    def scan_record(self) -> tuple[int, bytes] | None:
        """Read the next leveldb chunk header and payload."""
        if not self._opened_for_scan or self._fp is None:
            msg = "file not open for scanning"
            raise RuntimeError(msg)
        header = self._fp.read(LEVELDBLOG_HEADER_LEN)
        if len(header) == 0:
            return None
        if len(header) != LEVELDBLOG_HEADER_LEN:
            msg = f"record header is {len(header)} bytes, expected {LEVELDBLOG_HEADER_LEN}"
            raise ValueError(msg)
        checksum, dlength, dtype = struct.unpack("<IHB", header)
        self._index += LEVELDBLOG_HEADER_LEN
        data = self._fp.read(dlength)
        checksum_computed = zlib.crc32(data, self._crc[dtype]) & 0xFFFFFFFF
        if checksum != checksum_computed:
            msg = "record checksum is invalid, data may be corrupt"
            raise ValueError(msg)
        self._index += dlength
        return dtype, data

    def scan_data(self) -> bytes | None:
        """Return the next reassembled protobuf blob, or ``None`` at EOF."""
        if self._fp is None:
            msg = "file not open"
            raise RuntimeError(msg)
        offset = self._index % LEVELDBLOG_BLOCK_LEN
        space_left = LEVELDBLOG_BLOCK_LEN - offset
        if space_left < LEVELDBLOG_HEADER_LEN:
            pad_check = _strtobytes("\x00" * space_left)
            pad = self._fp.read(space_left)
            if pad != pad_check:
                msg = "invalid padding in transaction log block"
                raise ValueError(msg)
            self._index += space_left

        record = self.scan_record()
        if record is None:
            return None
        dtype, data = record
        if dtype == LEVELDBLOG_FULL:
            return data

        if dtype != LEVELDBLOG_FIRST:
            msg = f"expected record type {LEVELDBLOG_FIRST}, found {dtype}"
            raise ValueError(msg)
        while True:
            record = self.scan_record()
            if record is None:
                return None
            dtype, new_data = record
            if dtype == LEVELDBLOG_LAST:
                return data + new_data
            if dtype != LEVELDBLOG_MIDDLE:
                msg = f"expected record type {LEVELDBLOG_MIDDLE}, found {dtype}"
                raise ValueError(msg)
            data += new_data

    def _read_header(self) -> None:
        if self._fp is None:
            msg = "file not open"
            raise RuntimeError(msg)
        header = self._fp.read(LEVELDBLOG_HEADER_LEN)
        if len(header) != LEVELDBLOG_HEADER_LEN:
            msg = f"header is {len(header)} bytes, expected {LEVELDBLOG_HEADER_LEN}"
            raise ValueError(msg)
        ident, magic, version = struct.unpack("<4sHB", header)
        if ident != _strtobytes(LEVELDBLOG_HEADER_IDENT):
            msg = "invalid wandb transaction log ident"
            raise ValueError(msg)
        if magic != LEVELDBLOG_HEADER_MAGIC:
            msg = "invalid wandb transaction log magic"
            raise ValueError(msg)
        if version != LEVELDBLOG_HEADER_VERSION:
            msg = f"unsupported wandb transaction log version {version}"
            raise ValueError(msg)
        self._index += len(header)

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
