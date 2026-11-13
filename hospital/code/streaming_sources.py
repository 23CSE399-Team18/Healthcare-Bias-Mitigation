"""Read official gzip sources directly, including FAT32-safe segmented files.

The downloader stores a logical ``name.csv.gz`` as
``name.csv.gz.parts/manifest.json`` and numbered binary segments.  This module
joins compressed bytes in memory while reading, never into a large disk file.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import re


_HASH = re.compile(r"^[0-9a-f]{64}$")


class SourceIntegrityError(ValueError):
    """An incomplete, inconsistent, or changed source was encountered."""


def _manifest_path(source_path):
    source = Path(source_path)
    if source.name == "manifest.json":
        return source
    if source.is_dir() or source.name.endswith(".parts"):
        return source / "manifest.json"
    return Path(str(source) + ".parts") / "manifest.json"


def load_segment_manifest(source_path, verify=False):
    """Validate a complete manifest and segment sizes; optionally hash all bytes.

    ``verify=False`` trusts the downloader's recorded SHA256 after validating
    completion, expected/actual hash agreement, safe names, and current sizes.
    Use ``verify=True`` to rehash the stored compressed bytes independently.
    """
    manifest_path = _manifest_path(source_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SourceIntegrityError(f"No valid completed source manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != 1 or manifest.get("completed") is not True:
        raise SourceIntegrityError(f"Source download is incomplete: {manifest_path}")
    expected = manifest.get("expected_sha256", "")
    actual = manifest.get("sha256", "")
    if not _HASH.fullmatch(expected) or actual != expected:
        raise SourceIntegrityError("Recorded full-file hash does not match the publisher checksum")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise SourceIntegrityError("Source manifest contains no segments")
    total = 0
    full_digest = hashlib.sha256()
    for index, part in enumerate(parts):
        name = part.get("filename", "")
        if name != f"part-{index:05d}.bin":
            raise SourceIntegrityError("Unsafe, missing, or out-of-order segment filename")
        size = part.get("bytes")
        digest = part.get("sha256", "")
        if not isinstance(size, int) or size <= 0 or size >= 2**32 or not _HASH.fullmatch(digest):
            raise SourceIntegrityError("Invalid segment size or checksum")
        path = manifest_path.parent / name
        if not path.is_file() or path.stat().st_size != size:
            raise SourceIntegrityError(f"Missing or size-changed source segment: {path}")
        total += size
        if verify:
            part_digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    full_digest.update(chunk)
                    part_digest.update(chunk)
            if part_digest.hexdigest() != digest:
                raise SourceIntegrityError(f"Source segment SHA256 mismatch: {path}")
    if total != manifest.get("bytes"):
        raise SourceIntegrityError("Segment lengths do not equal the recorded source length")
    if verify and full_digest.hexdigest() != expected:
        raise SourceIntegrityError("Stored source SHA256 does not match the publisher checksum")
    return manifest_path, manifest


class SegmentedReader(io.RawIOBase):
    """Seekable binary view of a completed segmented source."""

    def __init__(self, source_path, verify=False):
        super().__init__()
        # RawIOBase can call close() while finalizing a failed constructor.
        self._stream = None
        self._stream_index = None
        manifest_path, self.manifest = load_segment_manifest(source_path, verify=verify)
        self._root = manifest_path.parent
        self._parts = self.manifest["parts"]
        self._starts = [0]
        for part in self._parts:
            self._starts.append(self._starts[-1] + part["bytes"])
        self._length = self._starts[-1]
        self._position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._length + offset
        else:
            raise ValueError("Invalid seek origin")
        if position < 0:
            raise ValueError("Cannot seek before the start of a source")
        self._position = position
        return position

    def readinto(self, buffer):
        self._checkClosed()
        target = memoryview(buffer).cast("B")
        copied = 0
        while copied < len(target) and self._position < self._length:
            index = bisect_right(self._starts, self._position) - 1
            if self._stream_index != index:
                if self._stream is not None:
                    self._stream.close()
                self._stream = (self._root / self._parts[index]["filename"]).open("rb")
                self._stream_index = index
            local_offset = self._position - self._starts[index]
            self._stream.seek(local_offset)
            available = min(len(target) - copied, self._starts[index + 1] - self._position)
            received = self._stream.readinto(target[copied:copied + available])
            if not received:
                raise SourceIntegrityError("Source segment ended before its recorded length")
            self._position += received
            copied += received
        return copied

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


@contextmanager
def open_compressed_binary(source_path, verify=False):
    """Open compressed bytes from a normal gzip file or its completed parts."""
    source = Path(source_path)
    if source.is_file() and source.name != "manifest.json":
        with source.open("rb") as stream:
            yield stream
    else:
        with SegmentedReader(source, verify=verify) as raw:
            with io.BufferedReader(raw, buffer_size=1024 * 1024) as stream:
                yield stream


@contextmanager
def open_csv_text(source_path, encoding="utf-8", verify=False):
    """Open logical .csv.gz source as text, without extracting the CSV to disk."""
    with open_compressed_binary(source_path, verify=verify) as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as uncompressed:
            with io.TextIOWrapper(uncompressed, encoding=encoding, newline="") as text:
                yield text


def read_official_csv(source_path, *, verify=False, **read_csv_kwargs):
    """Optional pandas convenience, preserving the stream while chunks are read.

    If ``chunksize`` or ``iterator=True`` is supplied, returns an iterator of
    DataFrames. Otherwise returns one DataFrame. Pandas is not required for the
    downloader or ``open_csv_text``.
    """
    import pandas as pd
    options = dict(read_csv_kwargs)
    options["compression"] = None
    if options.get("chunksize") is not None or options.get("iterator"):
        def chunks():
            with open_csv_text(source_path, verify=verify) as stream:
                with pd.read_csv(stream, **options) as reader:
                    yield from reader
        return chunks()
    with open_csv_text(source_path, verify=verify) as stream:
        return pd.read_csv(stream, **options)
