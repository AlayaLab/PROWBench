# Evaluation

The Python package `prowbench_eval` computes the metrics of the paper and builds the table rows.
[USAGE.md](USAGE.md) gives the commands, step by step.

## Tests

From `eval/`, with Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Tests that need the prepared data read it from `PROWBENCH_DATA` (default: `prowbench-eval-data/` at the repository root)
and are skipped without it. Installation of the model environments: [USAGE section 1](USAGE.md#1-environments).

## Metrics

| metric | definition |
|---|---|
| IoU | mean box IoU of the tracked entities against the recorded boxes (missing boxes count 0) |
| CErr | distance between box centers, normalized by the image size |
| Miss | share of instants at which an entity is not detected (multi-view) |
| Rot / Trans / CamMC | rotation error (degrees), translation error and CamMC of the estimated camera against the recorded camera |
| ISR | Interaction Success Rate: share of the events whose action and outcome the judge confirms |
| LRA | Logic-Render Alignment: share of the timeline segments whose action the judge confirms |
| gISR / gLRA | per-sample IoU x ISR / IoU x LRA |
| Img / Aes / Smooth | VBench imaging quality, aesthetic quality and motion smoothness |
| Warp | warp error between adjacent frames (x1000) |
| CLIP | CLIP similarity between the frames and the episode title (x100) |
| Comp. / Cons. | appearance compliance with the authored appearance / consistency across the views (multi-view, percent) |
| MEt3R | geometric consistency across the views (multi-view) |
| Re-IoU / Re-State | IoU after an entity returns into view / judged persistence of the walking or standing state of a returning person (long-horizon) |

A table cell is the mean over the evaluated views of one method; the multi-view table averages the four views of each scene
first, and the long-horizon table averages the ten recordings.
