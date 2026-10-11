"""Synthetic tests for release-input handling: ground-truth export options, CLIP title loading and the windowed
judge driver."""
import gzip
import json
import tempfile
import types
import unittest
from pathlib import Path

from prowbench_eval import gt_export, windowed_judging


def _gt(entities, frames, w=100, h=50):
    return dict(width=w, height=h, entities={e: dict(label="person") for e in entities},
                frames=[dict(frame=i, boxes=b) for i, b in enumerate(frames)])


class FirstPersonFilterTest(unittest.TestCase):
    def test_wearer_flagged_and_view_filling_entities_are_removed(self):
        full = dict(box=[0, 0, 100, 50], visible=True)
        small = dict(box=[10, 10, 20, 20], visible=True)
        frames = [dict(player=small, car=full, npc=small, bus=full if i < 9 else small) for i in range(10)]
        gt = _gt(["player", "car", "npc", "bus", "boat"], frames)
        self.assertEqual(gt_export.first_person_excluded(gt), ["bus", "car", "player"])
        self.assertEqual(gt_export.first_person_excluded(gt, flagged=["boat"]), ["boat", "bus", "car", "player"])
        out = gt_export.filter_first_person(gt)
        self.assertEqual(set(out["entities"]), {"npc", "boat"})
        self.assertTrue(all(set(f["boxes"]) == {"npc"} for f in out["frames"]))
        self.assertEqual(out["first_person_excluded"], ["bus", "car", "player"])
        self.assertIn("car", gt["entities"])  # input left unchanged

    def test_below_coverage_threshold_is_kept(self):
        full = dict(box=[0, 0, 100, 50], visible=True)
        small = dict(box=[10, 10, 20, 20], visible=True)
        frames = [dict(car=full if i < 8 else small) for i in range(10)]
        self.assertEqual(gt_export.first_person_excluded(_gt(["car"], frames)), [])


def _episode(root, ep, width, height, cameras):
    d = Path(root) / "episodes" / ep
    d.mkdir(parents=True)
    ent = dict(id="box", semanticLabel="crate")
    bbox = dict(min=[-1, -1, -1], max=[1, 1, 1], worldMatrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1])
    manifest = dict(fps=30, frameCount=1, width=width, height=height, states="frames.jsonl.gz", entities=[ent],
                    initial=dict(entities=dict(box=dict(bbox=bbox))), dynamicEntityIds=["box"], caseId=ep)
    (d / "manifest.json").write_text(json.dumps(manifest))
    with gzip.open(d / "frames.jsonl.gz", "wt") as f:
        f.write(json.dumps(dict(index=0, entities={})) + "\n")
    cam = dict(position=[0, 0, 0], quaternion=[0, 0, 0, 1], intrinsics=dict(fx=200.0, fy=200.0, cx=width / 2, cy=height / 2),
               worldToCamera=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    for c in cameras:
        (d / c).mkdir()
        (d / c / "camera.json").write_text(json.dumps(dict(frames=[cam])))
    return d


class GroundTruthExportTest(unittest.TestCase):
    def test_boxes_are_clipped_to_the_recording_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _episode(tmp, "ep", 40, 30, ["follow"])
            gt = gt_export.export_episode(d, "follow")
            box = gt["frames"][0]["boxes"]["box"]
            self.assertTrue(box["visible"])
            self.assertLessEqual(box["box"][2], 40.0)
            self.assertLessEqual(box["box"][3], 30.0)
            self.assertGreater(box["box_unclipped"][2], 40.0)

    def test_cli_discovers_cameras_and_filters_first_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            _episode(Path(tmp) / "ds", "ep", 40, 30, ["follow", "first-person"])
            gt_export.main(["--dataset", str(Path(tmp) / "ds"), "--out", str(Path(tmp) / "gt"), "--first-person-filter", "auto"])
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "gt" / "ep").iterdir()), ["first-person.json", "follow.json"])
            fp = json.loads((Path(tmp) / "gt" / "ep" / "first-person.json").read_text())
            self.assertEqual(fp["first_person_excluded"], ["box"])
            self.assertNotIn("first_person_excluded", json.loads((Path(tmp) / "gt" / "ep" / "follow.json").read_text()))
            gt_export.main(["--dataset", str(Path(tmp) / "ds"), "--out", str(Path(tmp) / "raw"), "--cameras", "first-person"])
            self.assertIn("box", json.loads((Path(tmp) / "raw" / "ep" / "first-person.json").read_text())["entities"])
            with self.assertRaises(SystemExit):
                gt_export.main(["--dataset", str(Path(tmp) / "ds"), "--out", str(Path(tmp) / "x"), "--cameras", "orbit"])


class ClipTitleTest(unittest.TestCase):
    def test_title_index_and_released_dataset(self):
        from prowbench_eval.metric_clipscore import load_titles
        with tempfile.TemporaryDirectory() as tmp:
            idx = Path(tmp) / "titles.json"
            idx.write_text(json.dumps(dict(episodes=[dict(episodeId="a", title="A person walks")])))
            self.assertEqual(load_titles(idx), {"a": "A person walks"})
            ds = Path(tmp) / "ds"
            (ds / "episodes" / "b").mkdir(parents=True)
            (ds / "episodes" / "b" / "scene-context.json").write_text(json.dumps(dict(title="riding a horse")))
            (ds / "index.json").write_text(json.dumps(dict(episodes=["b"])))
            self.assertEqual(load_titles(ds), {"b": "riding a horse"})
            self.assertEqual(load_titles(ds / "index.json"), {"b": "riding a horse"})
            (ds / "episodes" / "b" / "scene-context.json").write_text(json.dumps(dict(title=" ")))
            with self.assertRaises(ValueError):
                load_titles(ds)


def _desc(texts):
    return dict(fps=30, timeline=[dict(phase=f"p{i}", startSeconds=i, endSeconds=i + 1, description=t) for i, t in enumerate(texts)])


class WindowedJudgingTest(unittest.TestCase):
    def _descriptions(self, root, timelines):
        for ep, texts in timelines.items():
            p = Path(root) / "episodes" / ep / "follow" / "white"
            p.mkdir(parents=True)
            (p / "video.description.json").write_text(json.dumps(_desc(texts)))

    def test_foil_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._descriptions(tmp, {"s-001-city-walk": ["walks"], "s-002-city-walk": ["walks"],
                                     "s-003-park-run": ["walks"], "s-004-park-sit": ["sits down"]})
            s = dict(episode="s-001-city-walk", camera="follow", representation="colored-obb")
            self.assertEqual(windowed_judging.foil_for(tmp, s, "next_episode")[0], "s-003-park-run")
            self.assertEqual(windowed_judging.foil_for(tmp, s, "distinct_timeline")[0], "s-004-park-sit")
            self.assertEqual(windowed_judging.foil_for(tmp, s, "none"), (None, None))
            with self.assertRaises(ValueError):
                windowed_judging.foil_for(tmp, s, "other")

    def test_run_writes_results_and_retries_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sample = dict(sample_id="ep/follow/white", episode="ep", camera="follow", representation="white", method="m",
                          run_id="r", video="missing.mp4", fps=24, width=64, height=32, frame_count=120)
            (tmp / "samples.jsonl").write_text(json.dumps(sample) + "\n")
            (tmp / "gt" / "ep").mkdir(parents=True)
            (tmp / "gt" / "ep" / "follow.json").write_text(json.dumps(dict(width=64, height=32, frames=[])))
            (tmp / "protocol.json").write_text(json.dumps(dict(interaction=dict(frames=8, crop_margin=0.35))))
            (tmp / "registry.json").write_text(json.dumps(dict(instances=[])))
            (tmp / "events.json").write_text(json.dumps(dict(episodes=dict(ep=[]))))
            args = types.SimpleNamespace(metric="interaction", samples=tmp / "samples.jsonl", gt_dir=tmp / "gt",
                                         protocol=tmp / "protocol.json", out=tmp / "out", registry=tmp / "registry.json",
                                         visual_events=tmp / "events.json", descriptions=None, device="cpu",
                                         judge_cache=None, worker=0, workers=1)
            original = windowed_judging.interaction_sample
            try:
                windowed_judging.interaction_sample = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
                self.assertEqual(windowed_judging.run(args, judge=object()), (0, 1))
                out = json.loads((tmp / "out" / "ep__follow__white.interaction.json").read_text())
                self.assertEqual(out["validation_status"], "failed")
                calls = []
                windowed_judging.interaction_sample = lambda s, ev, gt, j, n, m: calls.append((n, m)) or dict(summary=dict(interaction_success=1.0))
                self.assertEqual(windowed_judging.run(args, judge=object()), (1, 0))
                self.assertEqual(calls, [(8, 0.35)])
                self.assertEqual(windowed_judging.run(args, judge=object()), (0, 0))  # kept, not re-judged
            finally:
                windowed_judging.interaction_sample = original


if __name__ == "__main__":
    unittest.main()
