import math
import unittest

from prowbench_eval import long_horizon_tables as lt


def _rec(iou, cerr, reappear, persist, n_persist, rot, trans, cammc, isr=None, lra=None, instants=None):
    return dict(
        entity=dict(summary=dict(mean_iou=iou, mean_center_err=cerr), per_entity=dict(a=dict(instants=instants or []))),
        memory=dict(summary=dict(reappear_iou=reappear)),
        state=dict(summary=dict(state_persist=persist, n_persist=n_persist)),
        camera=dict(summary=dict(RotErr_mean=rot, TransErr_rel_mean=trans, CamMC_rel_mean=cammc)),
        isr=isr, lra=lra)


class MethodRowTest(unittest.TestCase):
    def test_means_and_defined_only_columns(self):
        recs = [_rec(0.2, 0.05, None, None, 0, 0.1, 0.3, 0.6, isr=0.5, lra=1.0),
                _rec(0.4, 0.07, 0.2, 1.0, 3, 0.3, 0.5, 1.0, isr=1.0, lra=0.5)]
        row = lt.method_row(recs)
        self.assertAlmostEqual(row["iou"], 0.3)
        self.assertAlmostEqual(row["cerr"], 0.06)
        self.assertAlmostEqual(row["reappear_iou"], 0.2)          # undefined recording skipped
        self.assertEqual(row["n_reappear_recordings"], 1)
        self.assertAlmostEqual(row["state_persist"], 1.0)
        self.assertEqual(row["n_state_events"], 3)
        self.assertAlmostEqual(row["rot_deg"], math.degrees(0.2))
        self.assertAlmostEqual(row["trans"], 0.4)
        self.assertAlmostEqual(row["cammc"], 0.8)
        self.assertAlmostEqual(row["isr"], 0.75)
        self.assertAlmostEqual(row["gisr"], (0.2 * 0.5 + 0.4 * 1.0) / 2)
        self.assertAlmostEqual(row["glra"], (0.2 * 1.0 + 0.4 * 0.5) / 2)

    def test_missing_scores_leave_columns_out(self):
        recs = [_rec(0.2, 0.05, None, None, 0, 0.1, 0.3, 0.6)]
        recs[0]["camera"] = None
        row = lt.method_row(recs)
        self.assertNotIn("rot_deg", row)
        self.assertNotIn("gisr", row)
        self.assertIsNone(row["reappear_iou"])
        self.assertIsNone(row["state_persist"])

    def test_unmatched_recording_gets_the_center_error_penalty(self):
        recs = [_rec(0.0, None, None, None, 0, 0.1, 0.3, 0.6), _rec(0.4, 0.2, None, None, 0, 0.1, 0.3, 0.6),
                _rec(None, None, None, None, 0, 0.1, 0.3, 0.6)]          # last: no evaluated entity
        row = lt.method_row(recs)
        self.assertAlmostEqual(row["iou"], 0.2)
        self.assertAlmostEqual(row["cerr"], 0.6)                          # (1.0 + 0.2) / 2

    def test_partial_state_results_rejected(self):
        recs = [_rec(0.2, 0.05, None, 1.0, 1, 0.1, 0.3, 0.6), _rec(0.4, 0.07, None, 0.0, 1, 0.1, 0.3, 0.6)]
        recs[1]["state"] = None
        with self.assertRaises(ValueError):
            lt.method_row(recs)
        recs[0]["state"] = None
        self.assertIsNone(lt.method_row(recs)["state_persist"])

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            lt.method_row([])


class CurveAndCountsTest(unittest.TestCase):
    def test_window_curve_pools_instants_and_counts_misses_as_zero(self):
        a = [dict(t=0.0, gt=[0, 0, 1, 1], pred=[0, 0, 1, 1], iou=1.0), dict(t=4.9, gt=[0, 0, 1, 1], pred=None, iou=0.0),
             dict(t=5.0, gt=None), dict(t=9.0, gt=[0, 0, 1, 1], pred=[0, 0, 1, 1], iou=0.5)]
        b = [dict(t=1.0, gt=[0, 0, 1, 1], pred=[0, 0, 1, 1], iou=0.5)]
        res = [dict(per_entity=dict(x=dict(instants=a))), dict(per_entity=dict(y=dict(instants=b)))]
        curve = lt.window_iou_curve(res, window=5.0, duration=15.0)
        self.assertEqual(len(curve), 3)
        self.assertAlmostEqual(curve[0], (1.0 + 0.0 + 0.5) / 3)
        self.assertAlmostEqual(curve[1], 0.5)
        self.assertIsNone(curve[2])

    def test_scene_scores_and_count_below(self):
        rows = [dict(episode="e1", correct=True), dict(episode="e1", correct=False), dict(episode="e2", correct=True)]
        self.assertEqual(lt.scene_scores(rows), {"e1": 0.5, "e2": 1.0})
        with self.assertRaises(ValueError):
            lt.scene_scores([dict(episode="e", correct=True, method="m1"), dict(episode="e", correct=True, method="m2")])
        self.assertEqual(lt.count_below([0.1, 0.3, 0.29], 0.3), 2)
        self.assertIsNone(lt.mean_defined([None, None]))


if __name__ == "__main__":
    unittest.main()
