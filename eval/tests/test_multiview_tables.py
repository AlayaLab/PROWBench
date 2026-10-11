import json
import math
import unittest

import numpy as np

from prowbench_eval import camera_rectify, multiview_runner, multiview_tables

VIEWS = multiview_tables.VIEWS


def outputs(iou, cerr, miss, rot, trans, cammc, isr, lra):
    entity = dict(summary=dict(mean_iou=iou, mean_center_err=cerr, miss_rate=miss))
    camera = dict(protocol="cami2v_all_motion_v1", summary=dict(RotErr_mean=rot, TransErr_rel_mean=trans, CamMC_rel_mean=cammc))
    interaction = dict(summary=dict(interaction_success=isr, n_invalid=0), failures=[])
    alignment = dict(summary=dict(dynamic_checklist=lra, n_invalid=0), failures=[])
    return dict(entity=entity, camera=camera, interaction=interaction, alignment=alignment)


class MultiviewTablesTest(unittest.TestCase):
    def test_control_row_units(self):
        row = multiview_tables.control_row(
            dict(summary=dict(mean_iou=.5, mean_center_err=.1, miss_rate=.2)),
            dict(protocol="cami2v_all_motion_v1", summary=dict(RotErr_mean=math.pi / 180, TransErr_rel_mean=.3, CamMC_rel_mean=.4)))
        self.assertAlmostEqual(row["RotErr_deg"], 1.0)
        self.assertEqual(row["MissRate"], .2)
        with self.assertRaises(ValueError):
            multiview_tables.control_row(dict(summary={}), dict(protocol="ate", summary={}))

    def test_views_then_scenes(self):
        views = []
        for scene, base in [("s1", .2), ("s2", .6)]:
            for k, view in enumerate(VIEWS):
                views.append(dict(method="m", episode=scene, view=view,
                                  **outputs(base + .1 * k, .01, 0., 0., .1, .1, 1.0 if k else .5, .5)))
        result = multiview_tables.summarize_method(views, n_scenes=2)
        self.assertAlmostEqual(result["per_scene"]["s1"]["IoU"], .35)
        self.assertAlmostEqual(result["overall"]["IoU"], .55)
        # gISR is the mean of per-view products, not the product of means
        s1 = np.mean([.2 * .5, .3, .4, .5]); s2 = np.mean([.6 * .5, .7, .8, .9])
        self.assertAlmostEqual(result["overall"]["gISR"], (s1 + s2) / 2)
        self.assertAlmostEqual(result["overall"]["gLRA"], .5 * .55)

    def test_undefined_propagates(self):
        rows = [dict(method="m", episode="s", view=v, CenterErr=None if v == "follow-b" else .1) for v in VIEWS]
        out = multiview_tables.scene_macro(rows, ["CenterErr"])
        self.assertIsNone(out["per_scene"]["s"]["CenterErr"])
        self.assertIsNone(out["overall"]["CenterErr"])

    def test_requires_all_views_and_one_method(self):
        with self.assertRaises(ValueError):
            multiview_tables.scene_macro([dict(method="m", episode="s", view="follow-a", IoU=1.)], ["IoU"])
        rows = [dict(method=m, episode="s", view=v, IoU=1.) for m, v in zip("abab", VIEWS)]
        with self.assertRaises(ValueError):
            multiview_tables.scene_macro(rows, ["IoU"])

    def test_invalid_judgments_rejected(self):
        bad = dict(summary=dict(interaction_success=0., n_invalid=1), failures=[])
        with self.assertRaises(ValueError):
            multiview_tables.logic_row(bad, dict(summary=dict(dynamic_checklist=1., n_invalid=0), failures=[]))


class CameraRectifyTest(unittest.TestCase):
    def setUp(self):
        traj = [dict(intrinsics=dict(fx=f, fy=f, cx=640., cy=360.)) for f in np.linspace(1000, 1500, 30)]
        self.gt = dict(fps=30, frame_count=30, camera_trajectory=traj)
        self.sample = dict(width=640, height=360, crop_xywh=[0, 0, 1280, 720], fps=15, frame_count=15)

    def test_target_is_max_focal(self):
        ks, target = camera_rectify.per_frame_intrinsics(self.sample, self.gt)
        self.assertEqual(ks.shape, (15, 4))
        self.assertAlmostEqual(ks[0, 0], 500.)
        self.assertAlmostEqual(target[0], ks[:, 0].max())
        self.assertTrue(camera_rectify.needs_rectification(ks))
        gt = camera_rectify.rectified_ground_truth(self.sample, self.gt, target)
        self.assertEqual(len({json.dumps(f["intrinsics"]) for f in gt["camera_trajectory"]}), 1)
        self.assertAlmostEqual(gt["camera_trajectory"][0]["intrinsics"]["fx"], target[0] * 2)
        self.assertEqual(self.gt["camera_trajectory"][0]["intrinsics"]["fx"], 1000.)

    def test_constant_intrinsics_untouched(self):
        for f in self.gt["camera_trajectory"]:
            f["intrinsics"] = dict(fx=1000., fy=1000., cx=640., cy=360.)
        ks, _ = camera_rectify.per_frame_intrinsics(self.sample, self.gt)
        self.assertFalse(camera_rectify.needs_rectification(ks))

    def test_identity_and_center_crop(self):
        image = (np.arange(36 * 64 * 3) % 251).astype(np.uint8).reshape(36, 64, 3)
        k = np.array([50., 50., 32., 18.])
        np.testing.assert_array_equal(camera_rectify.rectify_frame(image, k, k), image)
        zoomed = camera_rectify.rectify_frame(image, k, np.array([100., 100., 32., 18.]))
        self.assertEqual(zoomed.shape, image.shape)
        self.assertEqual(int(zoomed[18, 32, 0]), int(image[18, 32, 0]))


class MultiviewRunnerTest(unittest.TestCase):
    def test_parse_and_score_compliance(self):
        job = dict(kind="prompt", method="m", episode="s", t=.5, participant="A", views=["follow-a"])
        raw = ('```json\n{"target":"present","attributes":{"clothing_color":"match","clothing_style":"mismatch",'
               '"hair":"unobservable","accessories":"unspecified","build":"match"},"reason":"x"}\n```')
        rows = multiview_runner.observation_rows(job, multiview_runner.parse_response(job, raw))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score"], 0.)
        self.assertAlmostEqual(rows[0]["attribute_score"], 2 / 3)

    def test_consistency_pairs_and_absent(self):
        job = dict(kind="consistency", method="m", episode="s", t=.5, participant="A", views=["follow-a", "follow-b", "follow-c"])
        ok = dict(clothing_color="consistent", clothing_style="consistent", hair="unobservable", accessories="unobservable", build="consistent")
        un = dict.fromkeys(ok, "unobservable")
        parsed = dict(targets={"follow-a": "present", "follow-b": "present", "follow-c": "absent"},
                      pairs=[dict(views=["follow-a", "follow-b"], attributes=ok, reason=""),
                             dict(views=["follow-a", "follow-c"], attributes=un, reason=""),
                             dict(views=["follow-b", "follow-c"], attributes=un, reason="")])
        rows = multiview_runner.observation_rows(job, multiview_runner.parse_response(job, json.dumps(parsed)))
        self.assertEqual([r["score"] for r in rows], [1., None, None])

    def test_rejects_invalid_response(self):
        job = dict(kind="prompt", method="m", episode="s", t=.5, participant="A", views=["follow-a"])
        with self.assertRaises(ValueError):
            multiview_runner.parse_response(job, '{"target":"present","attributes":{}}')
        with self.assertRaises(ValueError):
            multiview_runner.parse_response(job, "not json")

    def test_met3r_pairs(self):
        pairs = multiview_runner.met3r_pairs()
        self.assertEqual(len(pairs), 54)
        self.assertEqual(pairs[0], (.5, "follow-a", "follow-b"))


if __name__ == "__main__":
    unittest.main()


def test_outfits_fall_back_to_prompt_appearance():
    import json, tempfile
    from pathlib import Path
    from prowbench_eval import multiview_geometry as mg
    with tempfile.TemporaryDirectory() as tmp:
        ep = Path(tmp) / "episodes" / "ep"
        (ep).mkdir(parents=True)
        (ep / "manifest.json").write_text(json.dumps({"actors": [{"entityId": "player", "label": "A"}, {"entityId": "p-b", "label": "B"}]}))
        for view, text in [("follow-a", "The protagonist is participant A, a tall adult in a red coat. Next. <Subject 3> (participant B) is a short adult in blue, and keeps exactly this appearance."),
                           ("follow-b", "The protagonist is participant B, a short adult in blue. <Subject 3> (participant A) is not in frame at 00:00.000 and enters the frame at about 1.5 s; from then on it is a tall adult in a red coat, and keeps exactly this appearance.")]:
            (ep / view / "white").mkdir(parents=True)
            (ep / view / "engine.description.txt").write_text("Third-person follow view.")
            (ep / view / "white" / "video.h3-ref2va.txt").write_text(text)
        lab, look = mg.outfits(tmp, "ep", "follow-a")
        assert lab == {"player": "A", "p-b": "B"}
        assert look == {"A": "a tall adult in a red coat", "B": "a short adult in blue"}
        (ep / "follow-a" / "engine.description.txt").write_text("View.\nPeople:\nplayer: someone in green\n")
        assert mg.outfits(tmp, "ep", "follow-a")[1] == {"A": "someone in green"}
        (ep / "follow-b" / "white" / "video.h3-ref2va.txt").write_text("The protagonist is participant A, a different look. ")
        try:
            mg.prompt_outfits(tmp, "ep")
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent appearance accepted")


class ParticipantLabelsTest(unittest.TestCase):
    def test_labels_from_actors_or_multiview_order(self):
        from prowbench_eval.multiview_geometry import participant_labels
        released = dict(actors=[dict(entityId="player", label="A"), dict(entityId="participant-b", label="B")])
        self.assertEqual(participant_labels(released), {"player": "A", "participant-b": "B"})
        export = dict(name="s", multiview=dict(actors=["player", "participant-b", "participant-c", "participant-d"]))
        self.assertEqual(participant_labels(export), {"player": "A", "participant-b": "B", "participant-c": "C", "participant-d": "D"})
        for bad in ([], ["p", "p"], [f"p{i}" for i in range(27)]):
            with self.assertRaises(ValueError):
                participant_labels(dict(name="s", multiview=dict(actors=bad)))
        self.assertEqual(participant_labels(dict(multiview=dict(actors=[f"p{i}" for i in range(26)])))["p25"], "Z")
        both = dict(released, multiview=dict(actors=["participant-b", "player"]))
        self.assertEqual(participant_labels(both), {"player": "A", "participant-b": "B"})

    def test_outfits_of_every_released_view(self):
        import os
        from pathlib import Path
        from prowbench_eval import multiview_geometry
        data = Path(os.environ.get("PROWBENCH_DATA") or Path(__file__).resolve().parents[2] / "prowbench-eval-data") / "multi4"
        if not (data / "index.json").is_file():
            self.skipTest(f"no multi4 dataset under {data.parent} (set PROWBENCH_DATA)")
        views = sorted(p.parent for p in (data / "episodes").glob("*/*/camera.json"))
        self.assertEqual(len(views), 40)
        for view in views:
            lab, look = multiview_geometry.outfits(str(data), view.parent.name, view.name)
            self.assertEqual(sorted(lab.values()), ["A", "B", "C", "D"], view)
            self.assertTrue(look and set(look) <= set(lab.values()), (view, look))


class ClipFrameTest(unittest.TestCase):
    def test_any_frame_rate_uses_the_nearest_frame(self):
        import tempfile
        from pathlib import Path
        import cv2
        from prowbench_eval.multiview_geometry import Clip
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "v.mp4")
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 16))
            for i in range(125):
                out.write(np.full((16, 32, 3), (i % 20) * 12, np.uint8))
            out.release()
            clip = Clip(path)
            try:
                self.assertEqual(clip.fps, 25)
                self.assertAlmostEqual(float(clip.at(0.5).mean()), 12 * 12, delta=5)   # frame 12 (nearest, as round())
                self.assertAlmostEqual(float(clip.at(4.5).mean()), 12 * 12, delta=5)   # frame 112
            finally:
                clip.close()
