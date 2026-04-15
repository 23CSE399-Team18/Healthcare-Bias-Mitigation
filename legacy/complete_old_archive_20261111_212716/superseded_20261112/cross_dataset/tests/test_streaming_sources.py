"""Exercise segmented gzip boundaries, verification and authentication scoping."""
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streaming_sources as sources
import download_mimic as downloader


class SegmentedSourcesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="reader_test_", dir=ROOT)
        self.root = Path(self.temporary.name)
        self.csv_data = b"stay_id,value\n1,alpha\n2,beta\n3,gamma\n"
        self.compressed = gzip.compress(self.csv_data, mtime=0)
        self.logical = self.root / "test.csv.gz"
        self.parts = Path(str(self.logical) + ".parts")
        self.parts.mkdir()
        records = []
        # Split inside gzip headers, compressed blocks, and trailer, not only rows.
        for i, offset in enumerate(range(0, len(self.compressed), 7)):
            name = f"part-{i:05d}.bin"
            block = self.compressed[offset:offset + 7]
            (self.parts / name).write_bytes(block)
            records.append({"filename": name, "bytes": len(block),
                            "sha256": hashlib.sha256(block).hexdigest()})
        full_hash = hashlib.sha256(self.compressed).hexdigest()
        self.manifest = {"schema_version": 1, "completed": True,
                         "expected_sha256": full_hash, "sha256": full_hash,
                         "bytes": len(self.compressed), "parts": records}
        self.save_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def save_manifest(self):
        (self.parts / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_csv_reads_across_all_gzip_boundaries(self):
        with sources.open_csv_text(self.logical, verify=True) as stream:
            values = list(csv.DictReader(stream))
        self.assertEqual(values, [{"stay_id": "1", "value": "alpha"},
                                  {"stay_id": "2", "value": "beta"},
                                  {"stay_id": "3", "value": "gamma"}])

    def test_seekable_binary_exact_bytes(self):
        with sources.SegmentedReader(self.logical) as reader:
            self.assertEqual(reader.read(19), self.compressed[:19])
            self.assertEqual(reader.seek(-8, io.SEEK_END), len(self.compressed) - 8)
            self.assertEqual(reader.read(), self.compressed[-8:])
            reader.seek(2)
            self.assertEqual(reader.read(26), self.compressed[2:28])
            reader.seek(0)
            self.assertEqual(reader.read(), self.compressed)

    def test_regular_gzip_is_supported(self):
        self.logical.write_bytes(self.compressed)
        with sources.open_csv_text(self.logical) as stream:
            self.assertEqual(stream.read(), self.csv_data.decode())

    def test_incomplete_manifest_is_refused(self):
        self.manifest["completed"] = False
        self.save_manifest()
        with self.assertRaises(sources.SourceIntegrityError):
            with sources.open_csv_text(self.logical):
                pass

    def test_path_escape_is_refused(self):
        self.manifest["parts"][0]["filename"] = "../escape.bin"
        self.save_manifest()
        with self.assertRaises(sources.SourceIntegrityError):
            sources.load_segment_manifest(self.logical)

    def test_same_size_corruption_is_detected_when_rehashed(self):
        first = self.parts / "part-00000.bin"
        original = first.read_bytes()
        first.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaises(sources.SourceIntegrityError):
            sources.load_segment_manifest(self.logical, verify=True)

    def test_changed_size_is_always_refused(self):
        first = self.parts / "part-00000.bin"
        first.write_bytes(first.read_bytes()[:-1])
        with self.assertRaises(sources.SourceIntegrityError):
            sources.load_segment_manifest(self.logical)

    def test_full_hash_must_match_publisher(self):
        self.manifest["sha256"] = "f" * 64
        self.save_manifest()
        with self.assertRaises(sources.SourceIntegrityError):
            sources.load_segment_manifest(self.logical)


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


class FakeSession:
    def __init__(self, data, honor_range=True):
        self.data = data
        self.honor_range = honor_range
        self.offsets = []

    def open(self, url, *, offset=0):
        self.offsets.append(offset)
        if offset and self.honor_range:
            return FakeResponse(self.data[offset:], status=206,
                                headers={"Content-Range": f"bytes {offset}-{len(self.data)-1}/{len(self.data)}"})
        return FakeResponse(self.data)


class DownloaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="download_test_", dir=ROOT)
        self.root = Path(self.temporary.name)
        self.old_root = downloader.RAW_ROOT
        self.old_size = downloader.SEGMENT_BYTES
        downloader.RAW_ROOT = self.root
        downloader.SEGMENT_BYTES = 13
        self.name = "icu/chartevents.csv.gz"
        self.data = gzip.compress(b"stay_id,value\n1,alpha\n2,beta\n3,gamma\n", mtime=0)
        self.expected = hashlib.sha256(self.data).hexdigest()

    def tearDown(self):
        downloader.RAW_ROOT = self.old_root
        downloader.SEGMENT_BYTES = self.old_size
        self.temporary.cleanup()

    def make_prefix(self):
        directory = Path(str(self.root / self.name) + ".parts")
        directory.mkdir(parents=True)
        block = self.data[:13]
        (directory / "part-00000.bin").write_bytes(block)
        # An unfinished next chunk must be discarded rather than appended blindly.
        (directory / "part-00001.partial").write_bytes(b"unfinished")
        downloader.save_json(directory / "progress.json", {
            "url": downloader.BASE_URL + self.name, "expected_sha256": self.expected,
            "parts": [{"filename": "part-00000.bin", "bytes": 13,
                       "sha256": hashlib.sha256(block).hexdigest()}],
        })
        return directory

    def test_official_origin_scope(self):
        self.assertTrue(downloader.allowed_url(downloader.BASE_URL + self.name))
        for url in ["http://physionet.org/files/mimiciv/3.1/a", "https://evil.example/files/mimiciv/3.1/a",
                    "https://physionet.org/login/", "https://username:password@physionet.org/files/mimiciv/3.1/a",
                    "https://physionet.org:444/files/mimiciv/3.1/a",
                    "https://physionet.org/files/mimiciv/3.1/../../login/",
                    "https://physionet.org/files/mimiciv/3.1/%2e%2e/login/"]:
            self.assertFalse(downloader.allowed_url(url))

    def test_download_segments_verify_and_read(self):
        manifest = downloader.download_one(FakeSession(self.data), self.name, self.expected)
        self.assertTrue(manifest["completed"])
        self.assertTrue(all(p["bytes"] <= 13 for p in manifest["parts"]))
        with sources.open_csv_text(self.root / self.name, verify=True) as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_resume_uses_exact_completed_segment_offset(self):
        directory = self.make_prefix()
        session = FakeSession(self.data)
        downloader.download_one(session, self.name, self.expected)
        self.assertEqual(session.offsets, [13])
        self.assertFalse((directory / "part-00001.partial").exists())
        sources.load_segment_manifest(self.root / self.name, verify=True)

    def test_resume_rejects_ignored_range_without_appending(self):
        directory = self.make_prefix()
        with self.assertRaises(downloader.DownloadError):
            downloader.download_one(FakeSession(self.data, honor_range=False), self.name, self.expected)
        self.assertEqual((directory / "part-00000.bin").read_bytes(), self.data[:13])
        self.assertFalse((directory / "manifest.json").exists())
        self.assertFalse((directory / "part-00001.bin").exists())

    def test_publisher_hash_mismatch_is_never_completed(self):
        with self.assertRaises(downloader.DownloadError):
            downloader.download_one(FakeSession(self.data), self.name, "0" * 64)
        self.assertFalse((Path(str(self.root / self.name) + ".parts") / "manifest.json").exists())

    def test_html_cannot_be_a_publisher_manifest(self):
        with self.assertRaises(downloader.DownloadError):
            downloader.parse_publisher_checksums(b"<html>sign in</html>")

    def test_checksums_are_normalized_and_required(self):
        data = "\n".join("a" * 64 + "  ./" + name for name in downloader.FILES).encode()
        result = downloader.parse_publisher_checksums(data)
        self.assertEqual(result[self.name], "a" * 64)


if __name__ == "__main__":
    unittest.main()
