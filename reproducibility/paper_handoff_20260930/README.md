# Replay the three-context paper figure

This bundle reproduces the dated handoff's 13 selective-withdrawal confirmations
(52 withdrawal/control trajectories), then reintegrates the 12 trajectories in
the three-context figure. It includes the selected parameters, source and target
equilibrium records, independent equations, numerical checks, and renderer.
It does not need an existing `output/` directory or the running study.

From a fresh checkout, using Python 3.12 or later (verified with Python 3.14):

```sh
python3 -m venv /tmp/foi-paper-replay-venv
/tmp/foi-paper-replay-venv/bin/python -m pip install -r reproducibility/paper_handoff_20260930/requirements.txt
/tmp/foi-paper-replay-venv/bin/python scripts/replay_paper_handoff.py --output output/paper_handoff_replay
```

With the listed dependencies already installed, only the last command is needed,
using that environment's Python. The requirements pin the versions used for this
replay. Run without Python's `-O` option: the scripts use assertions for scientific
checks. The command refuses an existing output directory, including the original
handoff archive. A failed run leaves its partial output for inspection; retry in
a new directory.

Open `output/paper_handoff_replay/confirmed_withdrawal.html` locally. Its image,
vector figure, confirmation results, and source-record links are self-contained.
The directory also contains twelve CSV files (`time,E,I`), figure metadata, copies
of the replay scripts and inputs, and a SHA-256 manifest. The full calculation
includes independent 41-by-41 root searches at both inputs for all 13 contexts
and integrations through 10,000 ms of model time; it is more than a rendering step.

`inputs.json` records the bundled input hashes and their original archive hashes.
The JSON inputs and independent equation/confirmation scripts are byte-for-byte
copies from the 30 September handoff. The renderer changes only its link to the
unpublished full-study audit, replacing it with the bundled source records.
The equation script is a preserved historical source snapshot: the replay scripts
load its definitions before the original `results` block, avoiding that separate
historical experiment. The source-table and behavior hashes retained in the JSON
describe archive provenance; the original full tables are not bundled or claimed
to be re-audited by this replay.

This reproduces selected witnesses, not the full 74-case study or its partial
artifact audit. The state-role labels, parameters, time horizons, and numerical
criteria are those of the original handoff. Induction routes and neighborhood
extent remain separate measurements.
