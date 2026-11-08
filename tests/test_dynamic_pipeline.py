"""Fast checks for DVC verification, path safety, and dynamic data handling."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data.preprocess import preprocess_arabic
from src.data.split_data import split_data
from src.data.verify_dvc import verify_dataset
from src.pipeline.paths import PROCESSED_DIR, RAW_DIR, project_path, run_key


class DynamicPipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw_temp = TemporaryDirectory(dir=RAW_DIR)
        self.processed_temp = TemporaryDirectory(dir=PROCESSED_DIR)
        self.raw = Path(self.raw_temp.name)
        self.processed = Path(self.processed_temp.name)

    def tearDown(self):
        self.processed_temp.cleanup()
        self.raw_temp.cleanup()

    def make_dataset(self, *, labeled=True):
        rows = []
        for label in ("gender", "origin", "other", "religion"):
            for index in range(12):
                row = {"tweet": f"نص عربي {label} {index}"}
                if labeled:
                    row.update({
                        "target": label, "sentiment": "normal", "directness": "direct",
                        "group": "individual", "annotator_sentiment": "normal",
                    })
                rows.append(row)
        path = self.raw / "sample.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        checksum = hashlib.md5(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".dvc").write_text(
            f"outs:\n- md5: {checksum}\n  path: sample.csv\n", encoding="utf-8"
        )
        return path

    def test_dvc_detects_changed_dataset(self):
        path = self.make_dataset()
        self.assertEqual(verify_dataset(path), hashlib.md5(path.read_bytes()).hexdigest())
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed without updating DVC"):
            verify_dataset(path)

    def test_dvc_pointer_required(self):
        path = self.make_dataset()
        path.with_name(path.name + ".dvc").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "DVC metadata not found"):
            verify_dataset(path)

    def test_path_guard_and_run_isolation(self):
        path = self.make_dataset()
        self.assertEqual(project_path(path, RAW_DIR, file=True), path.resolve())
        with self.assertRaisesRegex(ValueError, "inside"):
            project_path(Path("data/raw") / ".." / ".." / "models" / "x.csv", RAW_DIR)
        self.assertNotEqual(run_key("manual__one"), run_key("manual__two"))
        self.assertEqual(len(run_key("manual__one")), 64)

    def test_train_preprocess_and_split_in_new_directory(self):
        path = self.make_dataset()
        ready = preprocess_arabic(path, self.processed)
        self.assertTrue({"tweet_clean", "target"} <= set(ready.columns))
        split_data(self.processed / "ar_ready.csv", self.processed)
        self.assertTrue(all((self.processed / name).is_file()
                            for name in ("train.csv", "val.csv", "test.csv")))

    def test_external_preprocess_without_labels(self):
        path = self.make_dataset(labeled=False)
        ready = preprocess_arabic(path, self.processed, mode="evaluate")
        self.assertEqual(len(ready), 48)
        self.assertNotIn("target", ready.columns)
        self.assertTrue((self.processed / "ar_ready.csv").is_file())

    def test_missing_training_columns_rejected(self):
        path = self.make_dataset(labeled=False)
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            preprocess_arabic(path, self.processed, mode="train")


if __name__ == "__main__":
    unittest.main()
