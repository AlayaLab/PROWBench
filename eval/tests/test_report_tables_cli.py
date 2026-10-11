import json
import tempfile
from pathlib import Path

from prowbench_eval import report_tables as rt


def test_rows_and_table_cli():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        m = tmp / "metrics"; m.mkdir()
        samples = [dict(sample_id=f"e{i}/follow/x", episode=f"e{i}", camera="follow", representation="x", method="m", run_id="r")
                   for i in range(2)]
        (tmp / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
        (m / "e0__follow__x.entity.json").write_text(json.dumps(dict(summary=dict(mean_iou=0.5, mean_center_err=0.1, n_entities=2))))
        (m / "e1__follow__x.entity.json").write_text(json.dumps(dict(summary=dict(mean_iou=0.0, mean_center_err=None, n_entities=1))))
        (m / "e0__follow__x.camera.json").write_text(json.dumps(dict(summary=dict(RotErr_mean=0.01, TransErr_rel_mean=0.2, CamMC_rel_mean=0.3))))
        vb = tmp / "vb.jsonl"
        vb.write_text(json.dumps({"sample_id": "e0/follow/x", "vbench.imaging_quality": 70.0, "vbench.aesthetic_quality": 0.6}) + "\n")
        rt.main(["rows", "--samples", str(tmp / "samples.jsonl"), "--metrics-dir", str(m), "--vbench", str(vb), "--out", str(tmp / "rows.jsonl")])
        rows = [json.loads(x) for x in (tmp / "rows.jsonl").read_text().splitlines()]
        assert rows[1]["cerr"] == 1.0 and rows[1]["center_error_penalty"] and rows[0]["imaging"] == 70.0 and rows[1]["imaging"] is None
        rt.main(["table", "--rows", str(tmp / "rows.jsonl"), "--out", str(tmp / "table.json")])
        cells = json.loads((tmp / "table.json").read_text())["m"]
        assert cells["iou"]["mean"] == 0.25 and cells["cerr"]["mean"] == 0.55 and cells["rot"]["n"] == 1


def test_table_expected_from_pairs_and_duplicates():
    rows = [dict(episode="a", camera="follow", n_entities=1, iou=0.5), dict(episode="b", camera="follow", n_entities=1, iou=0.7)]
    full = rt.table({"m": rows}, columns=("iou",), pairs=[("a", "follow"), ("b", "follow")])["m"]["iou"]
    assert full == dict(mean=0.6, n=2, expected=2)
    short = rt.table({"m": rows[:1]}, columns=("iou",), pairs=[("a", "follow"), ("b", "follow")])["m"]["iou"]
    assert short["n"] == 1 and short["expected"] == 2
    for bad, kw in [(rows + rows[:1], {}), (rows, dict(pairs=[("a", "follow")]))]:
        try:
            rt.table({"m": bad}, columns=("iou",), **kw)
        except ValueError:
            continue
        raise AssertionError("invalid rows accepted")
