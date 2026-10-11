"""Warp error over every generated frame pair, judge exit status and detect_track --force."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from prowbench_eval import detect_track, windowed_judging


def test_warp_error_uses_every_generated_pair_and_tof_only_at_equal_rate():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from prowbench_eval import metric_temporal
    def zero(model, a, b, iters=20):
        return torch.zeros(1, 2, *a.shape[2:])
    gen, ref = np.zeros((10, 4, 4, 3), np.uint8), np.zeros((5, 4, 4, 3), np.uint8)
    with patch.object(metric_temporal, "flow", zero):
        summary, rows = metric_temporal.evaluate(None, gen, ref, "cpu", 1, 60.0, 30.0)
        assert summary["n_pairs"] == 9 and summary["tof_pairs"] == 0 and summary["tof_epe"] is None
        assert [r["t"] for r in rows] == list(range(9))
        summary, _ = metric_temporal.evaluate(None, gen[:5], ref, "cpu", 1, 30.0, 30.0)
        assert summary["n_pairs"] == 4 and summary["tof_pairs"] == 4 and summary["tof_epe"] == 0.0
        summary, _ = metric_temporal.evaluate(None, gen[:3], ref, "cpu", 1, 16.0, 30.0)
        assert summary["n_pairs"] == 2 and summary["tof_pairs"] == 0


def test_windowed_judging_exits_non_zero_when_a_sample_fails():
    argv = ["alignment", "--samples", "s", "--gt-dir", "g", "--protocol", "p", "--descriptions", "d", "--out", "o"]
    with patch.object(windowed_judging, "run", return_value=(3, 1)):
        with pytest.raises(SystemExit) as exc:
            windowed_judging.main(argv)
        assert exc.value.code
    with patch.object(windowed_judging, "run", return_value=(4, 0)):
        windowed_judging.main(argv)


def test_detect_track_force_recomputes_every_job():
    with tempfile.TemporaryDirectory() as directory:
        jobs = Path(directory) / "jobs.json"
        jobs.write_text(json.dumps([dict(video="v.mp4", gt="g.json", out="o.json")]))
        for extra, expected in (([], None), (["--force"], True)):
            with patch.object(detect_track, "run_jobs") as run_jobs:
                detect_track.main(["--jobs", str(jobs)] + extra)
                assert run_jobs.call_args[0][0][0].get("force") is expected
