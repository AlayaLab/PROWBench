"""Long-horizon interaction and alignment scoring with optional adjudication."""
from collections import defaultdict
from statistics import mean
def adjudicated_interaction_success(action, outcome, action_verdict='u', outcome_verdict='u'):
    """Adjudicated y/n answers supersede judge answers; u retains the judge."""
    final = {}
    for key, judge, verdict in [('action', action, action_verdict), ('outcome', outcome, outcome_verdict)]:
        judge_value = {'yes': 'y', 'no': 'n'}.get(judge, 'n')
        final[key] = verdict if verdict in 'yn' else judge_value
    return final['action'] == 'y' and final['outcome'] == 'y'


def adjudicated_alignment_success(depicted, single_phase, action_verdict='u'):
    """Only a single-phase event's confident action verdict replaces the segment judgment."""
    return action_verdict == 'y' if single_phase and action_verdict in 'yn' else depicted == 'yes'


def scene_mean(rows):
    """Each row supplies episode and correct; average events within scenes first."""
    rows = list(rows)
    if len({(r.get('method'), r.get('kind')) for r in rows}) > 1:
        raise ValueError('Rows must belong to one method and one metric')
    scenes = defaultdict(list)
    for row in rows:
        scenes[row['episode']].append(bool(row['correct']))
    return mean(mean(values) for values in scenes.values())
