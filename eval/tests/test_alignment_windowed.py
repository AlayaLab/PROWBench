"""Tests of the LRA judging module `alignment_windowed`."""
from unittest.mock import patch
import numpy as np

from prowbench_eval import alignment_windowed as m

DESC = dict(fps=30, description="\n".join([
    "Generate a 5.00-second video of sitting on a chair and standing up.",
    "World: An open clearing at the edge of a coastal town has flat ground and two low buildings. Preserve the reference spacing. "
    "A single wooden chair with four legs stands on the ground.",
    "Protagonist: The protagonist is an adult wearing a muted olive shirt and navy trousers. Maintain the same face throughout the clip.",
    "Action timeline: From 00:00.000 to 00:00.767, the protagonist steps back. From 00:00.767 to 00:05.000, the protagonist sits. "
    "The chair remains in place and ends unoccupied.",
    "Camera: The camera makes a smooth arc of 95 degrees around the subject.",
    "Continuity: Keep world geometry stable."]),
    timeline=[dict(phase="step-back", startSeconds=0.0, endSeconds=0.767, description="The protagonist steps back."),
              dict(phase="sit", startFrame=23, endFrameExclusive=180, description="The protagonist sits."),
              dict(phase="blip", startSeconds=4.9, endSeconds=5.0, description="too short"),
              dict(phase="empty", startSeconds=1.0, endSeconds=2.0, description="")])
FOIL = dict(fps=30, description="World: A crate sits on the sand near the water.\nProtagonist: The protagonist wears a red jacket and boots.\n"
            "Action timeline: From 00:00.000 to 00:02.000, the protagonist pushes the crate. The crate ends displaced.",
            timeline=[dict(phase="push", startSeconds=0.0, endSeconds=2.0, description="The protagonist pushes the crate."),
                      dict(phase="release", startSeconds=2.0, endSeconds=5.0, description="The protagonist releases the crate.")])
SAMPLE = dict(sample_id="sit-stand/orbit/white", method="source", run_id="render", video="v.mp4", fps=30.0, frame_count=150,
              crop_xywh=[0, 0, 1280, 720])


def gt():
    frames = [dict(frame=i, time=i / 30, boxes={"player": dict(box=[600, 300, 640, 400], visible=True)}) for i in range(150)]
    return dict(width=1280, height=720, fps=30, frames=frames,
                entities={"player": dict(label="person", dynamic=True, interaction_candidate=True), "ground": dict(label="terrain", dynamic=False)})


def fake_frames(video, t0, t1, n, fps=None, max_side=768):
    idx = sorted({min(149, int(round(t * fps))) for t in np.linspace(t0, t1, n)})
    return [(i, np.zeros((72, 128, 3), np.uint8)) for i in idx]


class FakeJudge:
    model_name = "fake"

    def __init__(self, dyn=None, end=None, static=None, raise_on=None, static_foil=("yes", "no"), details=None, error=None):
        self.dyn, self.end, self.static, self.raise_on, self.static_foil = dyn or {}, end, static, raise_on or (), list(static_foil)
        self.error = error
        self.details = details or {}
        self.calls = []

    def ask(self, frames, prompt):
        self.calls.append(prompt)
        if "EXPECTED in this window" in prompt:
            text = prompt.split('EXPECTED in this window: "')[1].split('"')[0]
            if text in self.raise_on:
                from prowbench_eval.vlm_judge import JudgeResponseError
                raise (self.error or JudgeResponseError("judge answer is not JSON"))
            a = self.dyn.get(text, "no")
            return dict(parsed=dict(description="d", action=a, action_evidence=[frames[0][0], 999], details=self.details.get(text, a),
                                    details_evidence=[frames[0][0]], reason="r"))
        if "how the clip must END" in prompt:
            return dict(parsed=dict(description="d", answers=self.end, reason="r"))
        foil = "crate sits on the sand" in prompt
        return dict(parsed=dict(answers=self.static_foil if foil else self.static, reasons=["x"]))


def test_timeline_parsing_clips_to_five_seconds_and_drops_unusable_segments():
    segs = m.load_timeline(DESC)
    assert [s["phase"] for s in segs] == ["step-back", "sit"]
    assert segs[1]["t0"] == 23 / 30 and segs[1]["t1"] == 5.0


def test_prompt_facts_exclude_instructions_and_camera_and_find_end_state():
    facts = m.scene_facts(DESC)
    assert facts == ["An open clearing at the edge of a coastal town has flat ground and two low buildings.",
                     "A single wooden chair with four legs stands on the ground.",
                     "The protagonist is an adult wearing a muted olive shirt and navy trousers."]
    assert m.end_state_facts(DESC, m.load_timeline(DESC)) == ["The chair remains in place and ends unoccupied."]
    assert m.end_state_facts(dict(description="Action timeline: From 0 to 1, x.", timeline=[]), [dict(text="last seg")]) == ["last seg"]


def test_foil_is_deterministic_and_never_the_episode_itself():
    eps = ["zipline", "handoff", "push-crate", "sit-stand"]
    assert m.foil_episode(eps, "handoff") == "sit-stand" and m.foil_episode(eps, "zipline") == "push-crate"
    assert m.foil_episode(eps, "handoff", 1) == "push-crate" and m.foil_episode(eps, "zipline", 1) == "handoff"
    for bad in (0, 4):
        try:
            m.foil_episode(eps, "handoff", bad)
        except ValueError:
            continue
        raise AssertionError("foil offset mapping an episode onto itself was accepted")


def test_segment_prompt_carries_neighbour_context_and_strict_rate_needs_details():
    judge = FakeJudge(dyn={"The protagonist steps back.": "yes", "The protagonist sits.": "yes"}, details={"The protagonist sits.": "no"},
                      end=["yes"], static=["yes"])
    with patch.object(m, "sample_frames", fake_frames), patch.object(m, "sha256_file", lambda p: "sha"):
        r = m.evaluate_sample(SAMPLE, DESC, gt(), judge, dict(frames=4, static_frames=3, foil="none"))
    assert r["summary"]["dynamic_checklist"] == 1.0 and r["summary"]["dynamic_strict"] == 0.5
    first, second = judge.calls[0], judge.calls[1]
    assert 'Segment before this window: "(clip start)"' in first and 'Segment after this window: "The protagonist sits."' in first
    assert 'Segment before this window: "The protagonist steps back."' in second and 'Segment after this window: "(clip end)"' in second


def test_scoring_counts_yes_only_and_reports_unknown_invalid_and_margins():
    judge = FakeJudge(dyn={"The protagonist steps back.": "yes", "The protagonist sits.": "unknown", "The protagonist pushes the crate.": "yes"},
                      end=["yes"], static=["yes", "no", "maybe"])
    with patch.object(m, "sample_frames", fake_frames), patch.object(m, "sha256_file", lambda p: "sha"):
        r = m.evaluate_sample(SAMPLE, DESC, gt(), judge, dict(frames=4, static_frames=3), FOIL, "push-crate")
    s = r["summary"]
    assert s["dynamic_checklist"] == 0.5 and s["n_segments"] == 2 and s["dynamic_unknown_rate"] == 0.5
    assert s["end_state"] == 1.0 and s["n_end_facts"] == 1
    assert s["static_facts"] == 1 / 3 and s["n_static_facts"] == 3 and s["n_invalid"] == 1
    assert s["dynamic_foil"] == 0.5 and s["dynamic_margin"] == 0.0
    assert s["static_foil"] == 0.5 and abs(s["static_margin"] - (1 / 3 - 0.5)) < 1e-12
    assert r["segments"][0]["evidence"] == [0]
    assert r["foil"]["episode"] == "push-crate" and len(r["foil"]["segments"]) == 2
    assert all(int(t0 * 30) <= i <= int(t1 * 30) + 1 for seg in r["segments"] for (i, _) in seg["frames"] for t0, t1 in [seg["window"]])


def test_wrong_length_answers_and_judge_errors_are_invalid_not_zero():
    judge = FakeJudge(dyn={"The protagonist sits.": "yes"}, end=["yes", "yes"], static=["yes"], raise_on=("The protagonist steps back.",))
    with patch.object(m, "sample_frames", fake_frames), patch.object(m, "sha256_file", lambda p: "sha"):
        r = m.evaluate_sample(SAMPLE, DESC, gt(), judge, dict(frames=4, static_frames=3, foil="none"))
    s = r["summary"]
    assert r["segments"][0]["depicted"] == "invalid" and r["failures"][0]["question"] == "segment:step-back"
    assert s["dynamic_checklist"] == 0.5
    assert r["end_state"]["answers"] == ["invalid"]
    assert s["static_facts"] == 0.0 and s["n_invalid"] == 5 and r["foil"] is None


def test_judge_json_recovers_a_dropped_closing_brace_only():
    from prowbench_eval.vlm_judge import extract_json
    assert extract_json('```json\n{"answers": ["no", "no"], "reasons": ["a", "b"]\n```') == {"answers": ["no", "no"], "reasons": ["a", "b"]}
    assert extract_json('{"a": {"b": [1, 2]}') == {"a": {"b": [1, 2]}}
    assert extract_json('```json\n{"answers": ["yes"], "reasons": ["closed }] wrong"}]\n```') == {"answers": ["yes"], "reasons": ["closed }] wrong"]}
    try:
        extract_json('{"answers": ["no", "no"')
    except ValueError:
        return
    raise AssertionError("unterminated JSON was accepted")


def test_judge_execution_error_fails_the_sample():
    judge = FakeJudge(dyn={"The protagonist sits.": "yes"}, end=["yes", "yes"], static=["yes"], raise_on=("The protagonist steps back.",),
                      error=RuntimeError("CUDA out of memory"))
    with patch.object(m, "sample_frames", fake_frames), patch.object(m, "sha256_file", lambda p: "sha"):
        try:
            m.evaluate_sample(SAMPLE, DESC, gt(), judge, dict(frames=4, static_frames=3, foil="none"))
        except RuntimeError:
            return
    raise AssertionError("an execution error was recorded as an answer")
