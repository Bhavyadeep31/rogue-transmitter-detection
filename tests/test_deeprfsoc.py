import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rogue_detector.deeprfsoc import (
    _group_split_ids,
    _validate_and_summarize,
    import_deeprfsoc,
    train_deeprfsoc,
)


class DeepRFSoCTests(unittest.TestCase):
    def test_schema_layout_and_class_mapping(self):
        tiny = {
            ("BPSK", "-4"): np.zeros((2, 128, 96), dtype=np.float64),
            ("QPSK", "0"): np.ones((2, 128, 96), dtype=np.float64),
        }
        summary = _validate_and_summarize(tiny)
        self.assertEqual(summary["window_count"], 192)
        self.assertEqual(summary["sequence_length"], 128)
        self.assertEqual(summary["iq_layout"].split(";")[0], "(I/Q channel, time_sample, window_index)")
        self.assertEqual(summary["class_names"], ["BPSK", "QPSK"])

    def test_group_splits_keep_contiguous_blocks_together_and_are_reproducible(self):
        first = _group_split_ids(96, seed=77, group_size=16)
        again = _group_split_ids(96, seed=77, group_size=16)
        np.testing.assert_array_equal(first, again)
        self.assertEqual(set(first.tolist()), {0, 1, 2})
        for offset in range(0, 96, 16):
            self.assertEqual(np.unique(first[offset:offset + 16]).size, 1)

    def test_import_train_cuda_or_cpu_and_resume(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        temp_root = Path(__file__).parents[1] / "outputs" / "deeprfsoc"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            root = Path(temporary)
            raw = root / "tiny.pkl"
            rng = np.random.default_rng(5)
            data = {
                (modulation, str(snr)): rng.normal(size=(2, 128, 96)).astype(np.float64)
                for modulation in ("BPSK", "QPSK")
                for snr in (-4, 0)
            }
            with raw.open("wb") as stream:
                pickle.dump(data, stream)
            processed = root / "processed"
            import_deeprfsoc(str(raw), str(processed), seed=9, chunk_windows=24, split_group_windows=16)
            iq = np.load(processed / "iq.npy", mmap_mode="r")
            labels = np.load(processed / "labels.npy", mmap_mode="r")
            splits = np.load(processed / "split.npy", mmap_mode="r")
            self.assertEqual(iq.shape, (384, 2, 128))
            self.assertEqual(iq.dtype, np.float32)
            self.assertEqual(set(labels.tolist()), {0, 1})
            self.assertEqual(set(splits.tolist()), {0, 1, 2})
            # Each source cell contributes full contiguous groups to one split.
            cursor = 0
            for _cell in data:
                for start in range(cursor, cursor + 96, 16):
                    self.assertEqual(np.unique(splits[start:start + 16]).size, 1)
                cursor += 96
            latest = root / "latest.pt"
            model = root / "best.pt"
            metrics = root / "metrics.json"
            first_metrics = train_deeprfsoc(
                str(processed), str(model), str(metrics), epochs=1, batch_size=16,
                patience=3, seed=9, max_train_per_cell=16, max_eval_per_cell=8,
                latest_checkpoint_path=str(latest), device_name="cpu",
            )
            self.assertIn("test_accuracy", first_metrics)
            self.assertTrue(model.is_file())
            self.assertTrue(latest.is_file())
            resumed = train_deeprfsoc(
                str(processed), str(model), str(metrics), epochs=2, batch_size=16,
                patience=3, seed=9, max_train_per_cell=16, max_eval_per_cell=8,
                resume_from=str(latest), latest_checkpoint_path=str(latest), device_name="cpu",
            )
            self.assertIn("per_class", resumed)
            self.assertIn("accuracy_by_snr_db", resumed)
            self.assertTrue(metrics.is_file())
            if torch.cuda.is_available():
                gpu_metrics = train_deeprfsoc(
                    str(processed), str(root / "cuda.pt"), str(root / "cuda_metrics.json"),
                    epochs=1, batch_size=16, patience=2, seed=9,
                    max_train_per_cell=16, max_eval_per_cell=8,
                    latest_checkpoint_path=str(root / "cuda_latest.pt"),
                    device_name="cuda", precision="fp32",
                )
                self.assertIn("test_accuracy", gpu_metrics)
            for mapped in (iq, labels, splits):
                mapped._mmap.close()
            del iq, labels, splits


if __name__ == "__main__":
    unittest.main()
