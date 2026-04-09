"""Exercise the optional disease subset without accounts or network access."""
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import download_mimic as downloader
import import_browser_downloads as importer
import streaming_sources


class DiseaseDownloadProfileTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="disease_profile_test_", dir=ROOT)
        self.addCleanup(self.cleanup_temporary)
        self.root = Path(self.temporary.name)
        self.downloads = self.root / "Downloads"
        self.downloads.mkdir()
        self.raw = self.root / "raw"
        self.state = self.root / "browser_import_state.json"
        for name, value in (("RAW_ROOT", self.raw), ("STATE_FILE", self.state), ("SEGMENT_BYTES", 23)):
            active_patch = patch.object(importer, name, value)
            active_patch.start()
            self.addCleanup(active_patch.stop)
        self.payloads = {
            name: gzip.compress(f"synthetic_id,value\n{index},fixture\n".encode(), mtime=0)
            for index, name in enumerate(downloader.DISEASE_FILES)
        }
        self.checksums = {name: hashlib.sha256(data).hexdigest() for name, data in self.payloads.items()}
        self.write_manifest(self.downloads / "SHA256SUMS.txt", self.checksums)

    def cleanup_temporary(self):
        # FAT32 directory entries can remain busy briefly after file handles close.
        for attempt in range(3):
            try:
                self.temporary.cleanup()
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.2)

    @staticmethod
    def write_manifest(path, checksums):
        path.write_text("\n".join(f"{digest}  {name}" for name, digest in checksums.items()), encoding="ascii")

    def test_original_profiles_are_preserved(self):
        self.assertEqual(len(downloader.select_files()), 5)
        self.assertEqual(downloader.select_files(), downloader.FILES)
        self.assertEqual(downloader.select_files(metadata_only=True), downloader.FILES[:4])
        disease = downloader.select_files(disease_data=True)
        self.assertEqual(len(disease), 10)
        self.assertEqual(set(downloader.FILES) - set(disease), set())
        self.assertEqual(disease[-2:], ("icu/chartevents.csv.gz", "hosp/labevents.csv.gz"))
        with self.assertRaises(ValueError):
            downloader.select_files(disease_data=True, metadata_only=True)

    def test_missing_disease_checksum_is_refused(self):
        hashes = {name: self.checksums[name] for name in downloader.FILES}
        data = "\n".join(f"{digest}  {name}" for name, digest in hashes.items()).encode("ascii")
        downloader.parse_publisher_checksums(data)
        with self.assertRaisesRegex(downloader.DownloadError, "hosp/diagnoses_icd.csv.gz"):
            downloader.parse_publisher_checksums(data, downloader.DISEASE_FILES)

    def test_disease_import_verifies_all_ten_sources(self):
        originals = {}
        for name, data in self.payloads.items():
            path = self.downloads / Path(name).name
            path.write_bytes(data)
            originals[path] = data
        with redirect_stdout(io.StringIO()):
            state = importer.run_import(self.downloads, disease_data=True)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["profile"], "disease_data")
        self.assertEqual([record["relative_name"] for record in state["files"]], list(downloader.DISEASE_FILES))
        for name in downloader.DISEASE_FILES:
            _, manifest = streaming_sources.load_segment_manifest(self.raw / name, verify=True)
            self.assertEqual(manifest["sha256"], self.checksums[name])
            self.assertTrue(all(part["bytes"] <= 23 for part in manifest["parts"]))
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_default_import_still_needs_only_original_five(self):
        for name in downloader.FILES:
            (self.downloads / Path(name).name).write_bytes(self.payloads[name])
        with redirect_stdout(io.StringIO()):
            state = importer.run_import(self.downloads)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(state["files"]), 5)
        with redirect_stdout(io.StringIO()):
            expanded = importer.run_import(self.downloads, disease_data=True)
        self.assertEqual(expanded["status"], "waiting_for_files")
        self.assertEqual(set(expanded["missing_files"]), set(downloader.DISEASE_FILES) - set(downloader.FILES))

    def test_conflicting_disease_hash_in_browser_manifests_is_refused(self):
        changed = dict(self.checksums, **{"hosp/diagnoses_icd.csv.gz": "0" * 64})
        self.write_manifest(self.downloads / "SHA256SUMS (1).txt", changed)
        importer.choose_publisher_manifest(self.downloads)
        with self.assertRaisesRegex(importer.BrowserImportError, "different source versions"):
            importer.choose_publisher_manifest(self.downloads, downloader.DISEASE_FILES)

    def test_existing_saved_disease_hash_conflict_is_preserved(self):
        self.raw.mkdir()
        saved = self.raw / "SHA256SUMS.txt"
        changed = dict(self.checksums, **{"hosp/omr.csv.gz": "0" * 64})
        self.write_manifest(saved, changed)
        original = saved.read_bytes()
        with self.assertRaisesRegex(importer.BrowserImportError, "another version"):
            importer.run_import(self.downloads, disease_data=True)
        self.assertEqual(saved.read_bytes(), original)
        self.assertEqual(json.loads(self.state.read_text())["status"], "failed")
        self.assertEqual(list(self.raw.iterdir()), [saved])

    def test_existing_checksum_without_disease_entries_is_preserved(self):
        self.raw.mkdir()
        saved = self.raw / "SHA256SUMS.txt"
        self.write_manifest(saved, {name: self.checksums[name] for name in downloader.FILES})
        original = saved.read_bytes()
        with self.assertRaisesRegex(importer.BrowserImportError, "invalid for the selected files"):
            importer.run_import(self.downloads, disease_data=True)
        self.assertEqual(saved.read_bytes(), original)

    def test_downloader_cli_selects_disease_profile_before_downloading(self):
        state_path = self.root / "download_state.json"
        selected = []

        def fake_download(_session, name, expected):
            self.assertEqual(expected, self.checksums[name])
            selected.append(name)
            return {"bytes": len(self.payloads[name]), "sha256": expected}

        with patch.object(sys, "argv", ["download_mimic.py", "--disease-data"]), \
                patch.object(sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", return_value="synthetic_fixture"), \
                patch.object(downloader.getpass, "getpass", return_value="synthetic_fixture"), \
                patch.object(downloader, "STATE_FILE", state_path), \
                patch.object(downloader, "OfficialSession"), \
                patch.object(downloader, "fetch_publisher_manifest", return_value=self.checksums) as fetch, \
                patch.object(downloader, "download_one", side_effect=fake_download), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(downloader.main(), 0)
        self.assertEqual(selected, list(downloader.DISEASE_FILES))
        self.assertEqual(fetch.call_args.args[1], downloader.DISEASE_FILES)
        self.assertEqual(json.loads(state_path.read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
