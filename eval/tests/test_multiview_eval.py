import json
import tempfile
from argparse import Namespace
from pathlib import Path

from prowbench_eval import multiview_eval as mv

VIEWS = ("follow-a", "follow-b", "follow-c", "follow-d")


def _samples(tmp, method="m", crop=(0, 0, 1280, 720), views=VIEWS):
    rows = [dict(sample_id=f"ep/{v}/x", episode="ep", camera=v, representation="x", method=method, run_id="r",
                 video=f"{v}.mp4", fps=24, width=1280, height=720, frame_count=120, crop_xywh=list(crop)) for v in views]
    p = Path(tmp) / "samples.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _dataset(tmp, episodes=("ep",)):
    d = Path(tmp) / "dataset"
    d.mkdir(exist_ok=True)
    (d / "index.json").write_text(json.dumps(dict(episodes=list(episodes))))
    return str(d)


def test_load_samples_and_job_id():
    with tempfile.TemporaryDirectory() as tmp:
        method, media, crop, eps = mv.load_samples(_samples(tmp))
        assert method == "m" and eps == ["ep"] and crop == [0, 0, 1280, 720] and media["m"]["ep/follow-b"] == "follow-b.mp4"
        try:
            mv.load_samples(_samples(tmp, views=VIEWS[:3]))
        except SystemExit:
            pass
        else:
            raise AssertionError("missing view accepted")
    job = dict(kind="consistency", episode="ep", t=2.5, participant="B", views=["follow-a", "follow-c"])
    assert mv.job_id(job) == "consistency__ep__2.5__B__a-c"


def test_summarize_appearance_and_met3r():
    with tempfile.TemporaryDirectory() as tmp:
        sp = _samples(tmp)
        out = Path(tmp) / "out"
        (out / "appearance").mkdir(parents=True)
        (out / "met3r").mkdir()
        rows = [dict(kind="prompt", method="m", episode="ep", t=0.5, participant="A", views=["follow-a"], score=1.0),
                dict(kind="prompt", method="m", episode="ep", t=0.5, participant="A", views=["follow-b"], score=0.0),
                dict(kind="consistency", method="m", episode="ep", t=0.5, participant="A", views=["follow-a", "follow-b"], score=1.0)]
        for i, r in enumerate(rows):
            (out / "appearance" / f"j{i}.json").write_text(json.dumps(dict(job=dict(method="m"), rows=[r])))
        (out / "appearance" / "jobs__ep.json").write_text(json.dumps(dict(episode="ep", method="m", jobs=["j0", "j1", "j2"])))
        for t, a, b in mv.met3r_pairs():
            (out / "met3r" / f"ep__{t}__{a}__{b}.json").write_text(json.dumps(dict(method="m", episode="ep", t=t, views=[a, b], score=0.4, valid=True)))
        s = mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))
        assert s["compliance"]["percent"] == 50.0 and s["consistency"]["percent"] == 100.0
        assert abs(s["met3r"]["score"] - 0.4) < 1e-12 and s["met3r"]["pairs"] == s["met3r"]["expected_pairs"] == 54


def test_summarize_refuses_incomplete_results():
    with tempfile.TemporaryDirectory() as tmp:
        sp = _samples(tmp)
        out = Path(tmp) / "out"
        (out / "appearance").mkdir(parents=True)
        row = dict(kind="prompt", method="m", episode="ep", t=0.5, participant="A", views=["follow-a"], score=1.0)
        (out / "appearance" / "j0.json").write_text(json.dumps(dict(job=dict(method="m"), rows=[row])))
        (out / "appearance" / "jobs__ep.json").write_text(json.dumps(dict(episode="ep", method="m", jobs=["j0", "j1"])))
        s = mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))
        assert s["compliance"]["percent"] is None and s["n_missing"] == 1 and not s["appearance_complete"]
        (out / "appearance" / "j1.error.json").write_text("{}")
        (out / "appearance" / "j1.json").write_text(json.dumps(dict(job=dict(method="m"), rows=[row])))
        s = mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))
        assert s["compliance"]["percent"] is None and s["failed_judgments"] == ["j1.error.json"]
        (out / "appearance" / "j1.error.json").unlink()
        assert mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))["compliance"]["percent"] == 100.0
        assert mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))["met3r"]["score"] is None


def test_duplicate_views_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        p = _samples(tmp)
        p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n")
        try:
            mv.load_samples(p)
        except SystemExit:
            pass
        else:
            raise AssertionError("duplicate view accepted")


def test_met3r_rejects_extra_and_duplicate_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        sp = _samples(tmp)
        out = Path(tmp) / "out"
        (out / "met3r").mkdir(parents=True)
        for t, a, b in mv.met3r_pairs():
            (out / "met3r" / f"ep__{t}__{a}__{b}.json").write_text(json.dumps(dict(method="m", episode="ep", t=t, views=[a, b], score=0.4, valid=True)))
        assert abs(mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))["met3r"]["score"] - 0.4) < 1e-12
        t, a, b = mv.met3r_pairs()[0]
        (out / "met3r" / "copy.json").write_text(json.dumps(dict(method="m", episode="ep", t=t, views=[a, b], score=0.9, valid=True)))
        s = mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))["met3r"]
        assert s["score"] is None and s["duplicate_results"]
        (out / "met3r" / "copy.json").unlink()
        (out / "met3r" / "other.json").write_text(json.dumps(dict(method="m", episode="other", t=t, views=[a, b], score=0.9, valid=True)))
        s = mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp)))["met3r"]
        assert s["score"] is None and s["unexpected_results"] == ["other.json"]


def test_summarize_refuses_missing_scenes():
    with tempfile.TemporaryDirectory() as tmp:
        sp = _samples(tmp)
        out = Path(tmp) / "out"
        (out / "appearance").mkdir(parents=True)
        (out / "met3r").mkdir()
        row = dict(kind="prompt", method="m", episode="ep", t=0.5, participant="A", views=["follow-a"], score=1.0)
        (out / "appearance" / "j0.json").write_text(json.dumps(dict(job=dict(method="m"), rows=[row])))
        (out / "appearance" / "jobs__ep.json").write_text(json.dumps(dict(episode="ep", method="m", jobs=["j0"])))
        for t, a, b in mv.met3r_pairs():
            (out / "met3r" / f"ep__{t}__{a}__{b}.json").write_text(json.dumps(dict(method="m", episode="ep", t=t, views=[a, b], score=0.4, valid=True)))
        s = mv.summarize(Namespace(samples=str(sp), out=str(out), dataset=_dataset(tmp, ("ep", "ep2"))))
        assert s["missing_scenes"] == ["ep2"] and not s["appearance_complete"]
        assert s["compliance"]["percent"] is None and s["met3r"]["score"] is None
