"""Check browser-source verification and lossless FAT32 imports without network."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import import_browser_downloads as importer
import streaming_sources


class BrowserImportsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="browser_import_test_", dir=ROOT)
        self.root = Path(self.temporary.name)
        self.downloads = self.root / "Downloads"
        self.downloads.mkdir()
        self.old_raw = importer.RAW_ROOT
        self.old_state = importer.STATE_FILE
        self.old_size = importer.SEGMENT_BYTES
        importer.RAW_ROOT = self.root / "raw" / "mimiciv" / "3.1"
        importer.STATE_FILE = self.root / "browser_import_state.json"
        importer.SEGMENT_BYTES = 13
        self.payloads = {}
        self.expected = {}
        for index, relative in enumerate(importer.FILES):
            data = gzip.compress(f"synthetic_id,value\n{index},synthetic_fixture\n".encode(), mtime=0)
            self.payloads[relative] = data
            self.expected[relative] = hashlib.sha256(data).hexdigest()
        manifest = "\n".join(self.expected[name] + "  " + name for name in importer.FILES)
        (self.downloads / "SHA256SUMS.txt").write_text(manifest, encoding="ascii")

    def tearDown(self):
        importer.RAW_ROOT = self.old_raw
        importer.STATE_FILE = self.old_state
        importer.SEGMENT_BYTES = self.old_size
        self.temporary.cleanup()

    def place(self, relative, name=None):
        target = self.downloads / (name or Path(relative).name)
        target.write_bytes(self.payloads[relative])
        return target

    def test_hash_mismatch_is_refused_without_destination(self):
        relative = importer.FILES[0]
        (self.downloads / Path(relative).name).write_bytes(gzip.compress(b"wrong version"))
        with self.assertRaises(importer.BrowserImportError):
            importer.select_verified_source(self.downloads, relative, self.expected[relative])
        self.assertFalse(importer.RAW_ROOT.exists())

    def test_correct_duplicate_suffix_is_selected_and_segments_round_trip(self):
        relative = importer.FILES[0]
        canonical = self.downloads / Path(relative).name
        canonical.write_bytes(gzip.compress(b"wrong old download"))
        source_path = self.place(relative, "patients.csv (1).gz")
        source_original = source_path.read_bytes()
        source = importer.select_verified_source(self.downloads, relative, self.expected[relative])
        self.assertEqual(source["path"], source_path)
        manifest = importer.import_verified_source(relative, self.expected[relative], source)
        self.assertTrue(all(part["bytes"] <= 13 for part in manifest["parts"]))
        with streaming_sources.open_csv_text(importer.RAW_ROOT / relative, verify=True) as stream:
            self.assertEqual(list(csv.DictReader(stream)), [{"synthetic_id": "0", "value": "synthetic_fixture"}])
        self.assertEqual(source_path.read_bytes(), source_original)

    def test_incomplete_browser_file_is_never_selected(self):
        relative = importer.FILES[0]
        pending = self.downloads / (Path(relative).name + ".crdownload")
        pending.write_bytes(self.payloads[relative])
        self.assertIsNone(importer.select_verified_source(self.downloads, relative, self.expected[relative]))

    def test_change_after_verification_is_refused(self):
        relative = importer.FILES[0]
        path = self.place(relative)
        source = importer.select_verified_source(self.downloads, relative, self.expected[relative])
        path.write_bytes(gzip.compress(b"changed contents"))
        with self.assertRaises(importer.BrowserImportError):
            importer.import_verified_source(relative, self.expected[relative], source)
        self.assertFalse(Path(str(importer.RAW_ROOT / relative) + ".parts").exists())

    def test_file_change_during_hash_is_detected(self):
        path = self.place(importer.FILES[0])
        real_fstat = os.fstat
        calls = 0

        def changed_second_stat(descriptor):
            nonlocal calls
            calls += 1
            status = real_fstat(descriptor)
            if calls == 2:
                return SimpleNamespace(st_dev=status.st_dev, st_ino=status.st_ino,
                                       st_size=status.st_size, st_mtime_ns=status.st_mtime_ns + 1)
            return status

        with patch.object(importer.os, "fstat", side_effect=changed_second_stat):
            with self.assertRaises(importer.BrowserImportError):
                importer.hash_stable_file(path, require_gzip=True)

    def test_valid_existing_import_is_reused_without_overwrite(self):
        relative = importer.FILES[0]
        self.place(relative)
        source = importer.select_verified_source(self.downloads, relative, self.expected[relative])
        first = importer.import_verified_source(relative, self.expected[relative], source)
        manifest_path = Path(str(importer.RAW_ROOT / relative) + ".parts") / "manifest.json"
        before = manifest_path.stat().st_mtime_ns
        second = importer.import_verified_source(relative, self.expected[relative], source)
        self.assertEqual(first, second)
        self.assertEqual(before, manifest_path.stat().st_mtime_ns)
        with self.assertRaises(importer.BrowserImportError):
            importer.existing_verified_source(relative, "0" * 64)
        self.assertEqual(before, manifest_path.stat().st_mtime_ns)

    def test_all_five_files_complete_and_status_is_separate(self):
        for relative in importer.FILES:
            self.place(relative)
        terminal_status = self.root / "download_state.json"
        terminal_status.write_text('{"status":"access_required"}')
        state = importer.run_import(self.downloads)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(state["files"]), 5)
        self.assertFalse(state["credentials_read"])
        self.assertFalse(state["browser_profile_read"])
        self.assertEqual(json.loads(terminal_status.read_text())["status"], "access_required")
        for relative in importer.FILES:
            streaming_sources.load_segment_manifest(importer.RAW_ROOT / relative, verify=True)
            self.assertEqual((self.downloads / Path(relative).name).read_bytes(), self.payloads[relative])

    def test_completed_metadata_can_import_while_events_download(self):
        self.place(importer.FILES[0])
        state = importer.run_import(self.downloads)
        self.assertEqual(state["status"], "waiting_for_files")
        self.assertEqual(len(state["files"]), 1)
        self.assertEqual(state["missing_files"], list(importer.FILES[1:]))
        for relative in importer.FILES[1:]:
            self.place(relative)
        complete = importer.run_import(self.downloads)
        self.assertEqual(complete["status"], "complete")

    def test_partial_existing_destination_is_preserved(self):
        relative = importer.FILES[0]
        self.place(relative)
        directory = Path(str(importer.RAW_ROOT / relative) + ".parts")
        directory.mkdir(parents=True)
        earlier_partial = directory / "part-00000.partial"
        earlier_partial.write_bytes(b"previous incomplete download")
        with self.assertRaises(importer.BrowserImportError):
            importer.existing_verified_source(relative, self.expected[relative])
        self.assertEqual(earlier_partial.read_bytes(), b"previous incomplete download")


if __name__ == "__main__":
    unittest.main()
