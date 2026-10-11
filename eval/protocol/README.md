# Protocol files

`protocol/<dataset>/` holds, for each of the five datasets, the inputs used for the reported scores:

| file | content |
|---|---|
| `settings.json` | settings of every stage (judge model and decoding, tracker, camera estimation, quality metrics) |
| `events.json` | the events judged for the Interaction Success Rate (ISR): phases, participants, expected action and outcome |
| `descriptions.json` | the descriptions judged for the Logic-Render Alignment (LRA): the timeline sentences of each view |

The commands of [USAGE.md](../USAGE.md) pass these files to the judges (`--protocol`, `--visual-events`,
`--descriptions`); they are used as given.
