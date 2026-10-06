# Dated three-context paper handoff records

`inputs.json`, `source_contexts.json`, and `selective_screen_contexts.json`
preserve the selected parameters, original source records, hashes, and context
list from the 30 September 2026 handoff. They are dated evidence records, not
instructions for a fresh-checkout replay. The original independent equations,
confirmation, and rendering programs were Python. Their execution depended on
Python packages outside the Julia project and their source is retired from the
active repository. The `files` map in `inputs.json` is the original replay
manifest and names retired Python files; it does not describe the current
checkout's contents.

The original run's report, figures, checks, and logs remain under
`output/paper_handoff_20260930/` in the local archive. A copy of the retired
Python source is kept outside the Jujutsu working copies at
`retired_tools/python_sources_20260930/`; its three handoff program hashes
match `inputs.json`. These local materials are unavailable from a fresh
checkout, which cannot replay the original run from this bundle.

The recorded 13 context confirmations and 52 trajectories remain finite
historical observations. Reproducing them with a maintained Julia program
would require a separate implementation and numerical comparison; this bundle
does not claim to provide that program. The original source-table and behavior
hashes describe archive provenance. The full tables are not bundled here.
