"""Windowed interaction and timeline alignment for synchronized camera views."""
from .interaction_windowed import evaluate_sample as interaction
from .alignment_windowed import evaluate_sample as alignment


def evaluate_view(sample, events, ground_truth, description, foil_description, foil_id, judge, config):
    """Evaluate one view with eight interaction frames and a 0.35 crop margin."""
    return dict(interaction=interaction(sample, events, ground_truth, judge, n_frames=8, crop_margin=.35),
                alignment=alignment(sample, description, ground_truth, judge, config, foil_description, foil_id))
