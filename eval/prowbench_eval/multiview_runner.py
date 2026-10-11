"""Job construction and response handling for the multi-view appearance and MEt3R metrics.

Appearance Compliance: one judgment per participant, view and instant (0.5, 2.5 and 4.5 s), with the authored
appearance of that participant. Cross-view Consistency: one judgment per participant and instant over all views
in which the participant is visible (only when visible in at least two views), without the description.
The judge runs greedily with thinking disabled and at most 2400 new tokens; a response must be strict JSON
(optionally inside one Markdown code fence) that passes `multiview_appearance.validate`. Failed judgments are
errors, never absent or unobservable observations.
"""
import itertools
import json

from .multiview_appearance import CONSIST, PROMPT, panel, score_observation, validate
from .multiview_geometry import crops_for

APPEARANCE_TIMES = (0.5, 2.5, 4.5)
VIEWS = ("follow-a", "follow-b", "follow-c", "follow-d")
JUDGE_SETTINGS = dict(max_new_tokens=2400, enable_thinking=False)


def appearance_jobs(args, media, method, episode, cache, views=VIEWS, times=APPEARANCE_TIMES):
    """Judge jobs of one method and scene; each job carries its panels (BGR images) and its prompt.

    `args`, `media` and `cache` follow `multiview_geometry.crops_for`; `args.release` is the dataset root.
    """
    jobs = []
    for t in times:
        got = crops_for(args, media, method, episode, list(views), t, args.release, cache)
        for participant, per_view in sorted(got.items()):
            panels, description = [], None
            for view, (crop, frame, box, _is_subject, look) in sorted(per_view.items()):
                if not (participant in look and len(look[participant]) > 10):
                    raise ValueError('multiview_runner: check failed: participant in look and len(look[participant]) > 10')
                if description is not None:
                    if not (description == look[participant]):
                        raise ValueError('multiview_runner: check failed: description == look[participant]')
                description = look[participant]
                image = panel(crop, frame, box, view)
                panels.append((view, image))
                jobs.append(dict(kind="prompt", method=method, episode=episode, t=t, participant=participant,
                                 views=[view], images=[image], prompt=PROMPT.format(description=description)))
            if len(panels) >= 2:
                labels = [v for v, _ in panels]
                jobs.append(dict(kind="consistency", method=method, episode=episode, t=t, participant=participant,
                                 views=labels, images=[im for _, im in panels], prompt=CONSIST.format(views=", ".join(labels))))
    return jobs


def parse_response(job, raw):
    """Strict JSON of one judge response (an optional single code fence is removed); validated against the job."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if not (lines[0] in ("```json", "```") and lines[-1] == "```"):
            raise ValueError('multiview_runner: check failed: lines[0] in ("```json", "```") and lines[-1] == "```"')
        text = "\n".join(lines[1:-1])
    parsed = json.loads(text)
    validate(job, parsed)
    return parsed


def observation_rows(job, parsed):
    """Scored observations of one judgment: one row for a compliance job, one row per view pair for consistency."""
    base = {k: job[k] for k in ("kind", "method", "episode", "t", "participant")}
    if job["kind"] == "prompt":
        present = parsed["target"] == "present"
        return [dict(base, views=job["views"], attributes=parsed["attributes"],
                     **score_observation(parsed["attributes"], present=present))]
    rows = []
    for pair in parsed["pairs"]:
        present = all(parsed["targets"][view] == "present" for view in pair["views"])
        rows.append(dict(base, views=pair["views"], attributes=pair["attributes"],
                         **score_observation(pair["attributes"], present=present, consistency=True)))
    return rows


def met3r_pairs(views=VIEWS, times=None):
    """(t, view_a, view_b) for all six unordered pairs of the four views at the MEt3R instants."""
    from .multiview_met3r import CONFIG
    times = CONFIG["times"] if times is None else times
    return [(t, a, b) for t in times for a, b in itertools.combinations(views, 2)]
