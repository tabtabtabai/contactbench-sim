"""The contact-peak log goes to a gzipped sidecar; episode.json keeps the summary (no GPU)."""
import gzip
import json

from force_bench.g1.contactbench import CONTACT_LOG, split_contact_log


def test_the_per_frame_log_moves_to_the_sidecar(tmp_path):
    log = [{"t": i / 60, "finger": "f1", "N": 120.0 + i} for i in range(5000)]
    summary = {"peak": 300.0, "peak_normal": 250.0, "log_above_N": 100.0, "frames_above": 5000, "frames_above_logged": log}
    kept, name = split_contact_log(summary, tmp_path)
    assert name == CONTACT_LOG and "frames_above_logged" not in kept
    assert kept == {"peak": 300.0, "peak_normal": 250.0, "log_above_N": 100.0, "frames_above": 5000}
    assert len(json.dumps(kept)) < 200
    with gzip.open(tmp_path / CONTACT_LOG, "rt") as f:
        assert json.load(f)["frames_above_logged"] == log
