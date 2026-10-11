"""Tests of the shipped protocol inputs (protocol/<dataset>/): schema, consistency with the released data, and the loaders
of the windowed judges. Tests that read the released data use the directory named by PROWBENCH_DATA (default: prowbench-eval-data/ at the
repository root) and are skipped when it holds none of the datasets."""
import gzip
import math
import re
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from prowbench_eval import alignment_windowed, interaction_windowed, metric_clipscore, windowed_judging

EVAL = Path(__file__).resolve().parents[1]
PROTOCOL = EVAL / "protocol"
DATA = Path(os.environ.get("PROWBENCH_DATA") or EVAL.parent / "prowbench-eval-data")
DATASETS = ["verified_ff_40", "unverified_ff_90", "gunfight", "long_horizon", "multi4"]
TIMELINE_KEYS = ("phase", "startFrame", "endFrameExclusive", "startSeconds", "endSeconds", "description")


def _read(p):
    with open(p) as f:
        return json.load(f)


def _episodes(ds):
    return _read(DATA / ds / "index.json")["episodes"]


def _views(ds, ep):
    return sorted(p.parent.name for p in (DATA / ds / "episodes" / ep).glob("*/camera.json"))


def _phase_names(ds, ep):
    """phase names of the recording, plus name#k for the k-th run of each name"""
    names, runs, last = set(), {}, None
    with gzip.open(DATA / ds / "episodes" / ep / "frames.jsonl.gz", "rt") as f:
        for line in f:
            phase = (json.loads(line).get("interaction") or {}).get("phase")
            if phase is not None and phase != last:
                runs[str(phase)] = runs.get(str(phase), 0) + 1
                names |= {str(phase), f"{phase}#{runs[str(phase)]}"}
            last = phase
    return names


def _has_data(ds):
    return (DATA / ds / "index.json").is_file()


def _require_data(test):
    if not any(_has_data(ds) for ds in DATASETS):
        test.skipTest(f"no released dataset under {DATA} (set PROWBENCH_DATA)")


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


_VIEW_TEXT = {}


def _view_text(ds, ep, cam):
    """normalised released text of a view: prompts, descriptions, appearance, engine context and the scene context"""
    key = (ds, ep, cam)
    if key not in _VIEW_TEXT:
        d = DATA / ds / "episodes" / ep
        parts = []
        files = [p for p in (d / cam).rglob("*") if p.suffix in (".txt", ".json") and not ({"depth", "canonical-depth"} & set(p.parts))
                 and p.name not in ("camera.json", "visibility.json")]
        for p in sorted(files) + [d / "scene-context.json"]:
            if p.is_file():
                t = p.read_text()
                parts.append("\n".join(_strings(json.loads(t))) if p.suffix == ".json" else t)
        _VIEW_TEXT[key] = _norm("\n".join(parts))
    return _VIEW_TEXT[key]


def _released(sentence, text):
    return _norm(sentence).rstrip(".") in text


OUTCOME_PREFIX = "at the end of the segment this has just happened or is happening: "
LONG_FALLBACKS = {
    ("long_horizon", "long30-016-urban-park-revisit", "revisited", 800):
        ("The protagonist arrives back at the landmark and looks at it again.", "return_to_landmark"),
    ("long_horizon", "long30-017-home-camera", "finish", 863): ("The protagonist comes to a stop and stands still.", "stop"),
    ("long_horizon", "long30-019-warehouse-combat", "finish", 818): ("The protagonist comes to a stop and stands still.", "stop"),
}
_FRAMES = {}


def _frames(ds, ep):
    if (ds, ep) not in _FRAMES:
        with gzip.open(DATA / ds / "episodes" / ep / "frames.jsonl.gz", "rt") as f:
            _FRAMES[(ds, ep)] = [json.loads(line) for line in f]
    return _FRAMES[(ds, ep)]


def _speed(frames, fps, entity, a, b):
    pts = [f["entities"][entity]["position"] for f in frames[a:b]]
    return sum(math.dist((p[0], p[2]), (q[0], q[2])) for p, q in zip(pts, pts[1:])) / ((len(pts) - 1) / fps)


class SettingsTest(unittest.TestCase):
    def test_settings_of_every_dataset(self):
        self.assertTrue(DATASETS, "no protocol directories")
        for ds in DATASETS:
            s = _read(PROTOCOL / ds / "settings.json")
            self.assertEqual(s["dataset"], ds)
            self.assertEqual(s["judge"]["model"], "Qwen/Qwen3.6-27B")
            self.assertEqual(s["judge"]["max_new_tokens"], 2048)
            self.assertIs(s["judge"]["enable_thinking"], False)
            self.assertEqual((s["interaction"]["frames"], s["interaction"]["crop_margin"]), (8, 0.35))
            a = s["alignment"]
            self.assertEqual((a["frames"], a["window_pad"], a["crop_margin"], a["seed"], a["foil"]), (12, 0.3, 0.35, 0, "distinct_timeline"))
            self.assertEqual(s["files"], {"events": "events.json", "descriptions": "descriptions.json"})
            for name in s["files"].values():
                self.assertTrue((PROTOCOL / ds / name).is_file(), f"{ds}/{name}")
            if _has_data(ds):
                self.assertEqual(a["foil_offset"], len(_episodes(ds)) // 2)


class EventsTest(unittest.TestCase):
    def test_event_groups_refer_to_released_entities_and_phases(self):
        _require_data(self)
        for ds in DATASETS:
            p = PROTOCOL / ds / "events.json"
            if not p.is_file() or not _has_data(ds):
                continue
            table = _read(p)
            self.assertEqual(table["dataset"], ds)
            self.assertIsInstance(table["window_pad_seconds"], float)
            self.assertEqual(sorted(table["episodes"]), sorted(_episodes(ds)))
            for ep, groups in table["episodes"].items():
                ids = {e["id"] for e in _read(DATA / ds / "episodes" / ep / "manifest.json")["entities"]}
                phases = _phase_names(ds, ep)
                self.assertTrue(groups, ep)
                for g in groups:
                    self.assertTrue({"name", "phases", "participants", "action", "outcome"} <= set(g), (ep, g))
                    self.assertTrue(set(g["participants"]) <= ids, (ep, g["name"]))
                    self.assertTrue(set(g["phases"]) <= phases, (ep, g["name"]))
                    if "camera" in g:
                        self.assertIn(g["camera"], _views(ds, ep))


class DescriptionsTest(unittest.TestCase):
    def test_descriptions_cover_every_view_and_match_the_engine_timeline(self):
        _require_data(self)
        for ds in DATASETS:
            p = PROTOCOL / ds / "descriptions.json"
            if not p.is_file() or not _has_data(ds):
                continue
            table = _read(p)
            self.assertEqual(table["dataset"], ds)
            self.assertEqual(sorted(table["episodes"]), sorted(_episodes(ds)))
            cameras = {c for ep in _episodes(ds) for c in _views(ds, ep)}
            for ep, records in table["episodes"].items():
                views = _views(ds, ep)
                self.assertTrue(set(views) <= set(records), ep)
                for cam, rec in records.items():
                    self.assertIn(cam, cameras)
                    self.assertEqual(rec.get("view_in_dataset", True), cam in views, (ep, cam))
                    for t in rec["timeline"]:
                        self.assertEqual(tuple(t), TIMELINE_KEYS)
                    segs = alignment_windowed.load_timeline(rec)
                    self.assertTrue(alignment_windowed.scene_facts(rec))
                    self.assertTrue(alignment_windowed.end_state_facts(rec, segs))
                    ctx = DATA / ds / "episodes" / ep / cam / "engine.context.json"
                    if ctx.is_file():
                        engine = _read(ctx)["timeline"]
                        self.assertEqual([{k: t[k] for k in TIMELINE_KEYS} for t in engine],
                                         [dict(t, description=windowed_judging._FP_NOTE.sub("", t["description"]).strip())
                                          for t in rec["timeline"]], (ep, cam))

    def test_file_and_directory_layouts_give_the_same_records_and_foils(self):
        _require_data(self)
        for ds in DATASETS:
            p = PROTOCOL / ds / "descriptions.json"
            if not p.is_file() or not _has_data(ds):
                continue
            table = _read(p)["episodes"]
            with tempfile.TemporaryDirectory() as tmp:
                for ep, records in table.items():
                    for cam, rec in records.items():
                        d = Path(tmp) / "episodes" / ep / cam / "white"
                        d.mkdir(parents=True)
                        (d / "video.description.json").write_text(json.dumps(rec))
                self.assertEqual(windowed_judging.description_episodes(p), windowed_judging.description_episodes(tmp))
                self.assertEqual(windowed_judging.description_episodes(p), sorted(table))
                offset = _read(PROTOCOL / ds / "settings.json")["alignment"]["foil_offset"]
                for ep in _episodes(ds):
                    for cam in _views(ds, ep):
                        sample = dict(episode=ep, camera=cam, representation="colored-obb")
                        self.assertEqual(windowed_judging.read_description(p, sample), windowed_judging.read_description(tmp, sample))
                        f1 = windowed_judging.foil_for(p, sample, "distinct_timeline", offset)
                        f2 = windowed_judging.foil_for(tmp, sample, "distinct_timeline", offset)
                        self.assertEqual(f1, f2)
                        self.assertNotEqual(f1[0], ep)


class ReleasedTextTest(unittest.TestCase):
    """Every text of the protocol files is the released text of the view it belongs to."""

    def test_description_text_occurs_in_the_released_view(self):
        _require_data(self)
        para = re.compile(r"^\s*([A-Z][A-Za-z ]{2,30}):\s*(.*\S)\s*$")
        for ds in DATASETS:
            p = PROTOCOL / ds / "descriptions.json"
            if not p.is_file() or not _has_data(ds):
                continue
            for ep, records in _read(p)["episodes"].items():
                ref = _views(ds, ep)[0]
                for cam, rec in records.items():
                    text = _view_text(ds, ep, cam if cam in _views(ds, ep) else ref)
                    for t in rec["timeline"]:
                        s = windowed_judging._FP_NOTE.sub("", t["description"]).strip()
                        if not _released(s, text):
                            self._check_fallback(ds, ep, t, s)
                    for line in rec["description"].split("\n"):
                        m = para.match(line)
                        if m and m.group(1) == "Action timeline" or not line.strip() or line.strip() == "People:":
                            continue
                        body = m.group(2) if m else line
                        if not m and re.match(r"^[a-z0-9-]+: ", line):
                            body = line.split(": ", 1)[1]
                        for s in alignment_windowed.split_sentences(body):
                            self.assertTrue(_released(s, text), (ds, ep, cam, s))

    def _check_fallback(self, ds, ep, t, sentence):
        """long-horizon phases at the end of a recording that no window prompt names: a fixed sentence, listed here with
        the recorded evidence it states"""
        key = (ds, ep, t["phase"], t["startFrame"])
        self.assertIn(key, LONG_FALLBACKS, (key, sentence))
        expected, evidence = LONG_FALLBACKS[key]
        self.assertEqual(sentence, expected)
        frames = _frames(ds, ep)
        fps = _read(DATA / ds / "episodes" / ep / "manifest.json")["fps"]
        a, b = t["startFrame"], t["endFrameExclusive"]
        self.assertLess(_speed(frames, fps, "player", max(a, b - 15), b), 0.15, key)
        if evidence == "return_to_landmark":
            m = _read(DATA / ds / "episodes" / ep / "manifest.json")["initial"]["entities"]["revisit-landmark"]["bbox"]["worldMatrix"]
            def dist(f):
                return math.dist((f["entities"]["player"]["position"][0], f["entities"]["player"]["position"][2]), (m[12], m[14]))
            self.assertAlmostEqual(dist(frames[a]), dist(frames[0]), delta=0.1)

    def test_event_text_is_built_from_the_timeline_of_its_phases(self):
        for ds in DATASETS:
            desc = _read(PROTOCOL / ds / "descriptions.json")["episodes"]
            for ep, groups in _read(PROTOCOL / ds / "events.json")["episodes"].items():
                for g in groups:
                    cam = g.get("camera") or sorted(c for c, r in desc[ep].items() if r.get("view_in_dataset", True))[0]
                    seq = [(t["phase"], windowed_judging._FP_NOTE.sub("", t["description"]).strip()) for t in desc[ep][cam]["timeline"]]
                    names = [p for p, _ in seq]
                    idx = [len(names) - 1 - names[::-1].index(p) for p in g["phases"]]
                    action = re.sub(r"^\(First-person view:.*?\)\s*", "", g["action"])
                    self.assertEqual(action, " ".join(seq[i][1] for i in idx), (ds, ep, g["name"]))
                    nxt = max(idx) + 1
                    self.assertEqual(g["outcome"], OUTCOME_PREFIX + seq[nxt if nxt < len(seq) else max(idx)][1], (ds, ep, g["name"]))

    def test_action_timeline_paragraph_follows_the_timeline(self):
        para = re.compile(r"^Action timeline: (.*)$", re.M)
        for ds in DATASETS:
            for ep, records in _read(PROTOCOL / ds / "descriptions.json")["episodes"].items():
                for cam, rec in records.items():
                    m = para.search(rec["description"])
                    if not m:
                        continue
                    expected = " ".join(f"From {t['startSeconds']:.2f} to {t['endSeconds']:.2f} s: "
                                        + windowed_judging._FP_NOTE.sub("", t["description"]).strip() for t in rec["timeline"])
                    self.assertEqual(m.group(1), expected, (ds, ep, cam))

    def test_clip_titles_are_the_scene_context_titles(self):
        _require_data(self)
        for ds in DATASETS:
            if not _has_data(ds) or "quality" not in _read(PROTOCOL / ds / "settings.json"):
                continue
            titles = metric_clipscore.load_titles(DATA / ds)
            self.assertEqual(sorted(titles), sorted(_episodes(ds)))
            for ep, t in titles.items():
                self.assertEqual(t, _read(DATA / ds / "episodes" / ep / "scene-context.json")["title"])


class LoaderTest(unittest.TestCase):
    def _record(self, text):
        return dict(fps=30, timeline=[dict(phase="walk", startFrame=0, endFrameExclusive=150, startSeconds=0.0, endSeconds=5.0,
                                           description=text)],
                    description="World: A quiet street lined with tall trees.\nAction timeline: From 0.00 to 5.00 s: " + text)

    def test_descriptions_file_lookup_missing_camera_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "descriptions.json"
            p.write_text(json.dumps(dict(episodes={"b": {"follow": self._record("B walks.")}, "a": {"follow": self._record("A walks.")}})))
            s = dict(episode="a", camera="follow", representation="white")
            self.assertEqual(windowed_judging.description_episodes(p), ["a", "b"])
            self.assertEqual(windowed_judging.read_description(p, s)["timeline"][0]["description"], "A walks.")
            self.assertEqual(windowed_judging.read_description(p, s, "b")["timeline"][0]["description"], "B walks.")
            with self.assertRaises(KeyError):
                windowed_judging.read_description(p, dict(s, camera="first-person"))
            time.sleep(0.01)
            p.write_text(json.dumps(dict(episodes={"a": {"follow": self._record("A runs.")}})))
            os.utime(p, None)
            self.assertEqual(windowed_judging.read_description(p, s)["timeline"][0]["description"], "A runs.")
            self.assertEqual(windowed_judging.description_episodes(p), ["a"])

    def test_event_groups_take_their_windows_from_the_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.json"
            p.write_text(json.dumps(dict(window_pad_seconds=0.15, episodes={"ep": [
                dict(name="walk-stop", phases=["walk", "stop"], participants=["player"], action="A.", outcome="B.")]})))
            reg = dict(instances=[dict(kind="phase", episode="ep", name="walk", event_id="e1", start_time=0.0, end_time=2.0),
                                  dict(kind="phase", episode="ep", name="stop", event_id="e2", start_time=2.0, end_time=5.0),
                                  dict(kind="state_change", episode="ep", name="x", event_id="e3", start_time=1.0, end_time=1.1)])
            (ev,) = interaction_windowed.load_visual_events(str(p), "ep", reg)
            self.assertEqual((ev["start"], ev["end"], ev["window"]), (0.0, 5.0, (0.0, 5.15)))
            self.assertEqual(ev["phase_ids"], ["e1", "e2"])

    def test_repeated_phase_names_are_addressed_by_occurrence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.json"
            p.write_text(json.dumps(dict(window_pad_seconds=0.0, episodes={"ep": [
                dict(name="first", phases=["walk#1"], participants=["player"], action="A.", outcome="B."),
                dict(name="second", phases=["walk#2", "turn"], participants=["player"], action="A.", outcome="B."),
                dict(name="plain", phases=["walk"], participants=["player"], action="A.", outcome="B.")]})))
            reg = dict(instances=[dict(kind="phase", episode="ep", name="turn", event_id="t1", start_time=2.0, end_time=3.0),
                                  dict(kind="phase", episode="ep", name="walk", event_id="w1", start_time=0.0, end_time=2.0),
                                  dict(kind="phase", episode="ep", name="walk", event_id="w2", start_time=3.0, end_time=5.0),
                                  dict(kind="phase", episode="other", name="walk", event_id="o1", start_time=0.0, end_time=5.0)])
            first, second, plain = interaction_windowed.load_visual_events(str(p), "ep", reg)
            self.assertEqual((first["window"], first["phase_ids"]), ((0.0, 2.0), ["w1"]))
            self.assertEqual((second["window"], second["phase_ids"]), ((2.0, 5.0), ["w2", "t1"]))
            self.assertEqual(plain["phase_ids"], ["w2"])
            missing = Path(tmp) / "missing.json"
            missing.write_text(json.dumps(dict(episodes={"ep": [dict(name="x", phases=["walk#3"], participants=[], action="A.",
                                                                     outcome="B.")]})))
            with self.assertRaises(ValueError):
                interaction_windowed.load_visual_events(str(missing), "ep", reg)


if __name__ == "__main__":
    unittest.main()
