# PathVeil project conventions

- The formal entry point is `scripts/run_experiment.sh YYYYMMDD`.
- Keep numerical experiment Python in `src/pathveil/`; shell scripts are thin entry points.
- Resource dispatchers may live in `scripts/` and must call the frozen numerical APIs, preserve input/model/code identities, and reuse verified caches.
- Do not restore old LLM review, voting pipelines, duplicate launchers or trial directories.
- Keep large datasets in `data/`, trained assets in `models/`, outputs in `runs/`.
- Register each experiment date in `configs/dates/`. Never select a nearby snapshot implicitly.
- Public evidence, training origins, held-out sources and final route labels have distinct roles.
- Training labels and PG local knowledge must consume the same admitted library and verify its path-set hash.
- Preserve `unknown` relationships. Observation support does not prove forwarding reality or downstream accuracy.
- Cache identities include method/config, input, model, date and feature schema hashes.
- Checkpoints use BGP-derived pair-disjoint validation; final held-out labels are used only for evaluation.
- New exploratory code belongs in an explicit branch or the separate legacy workspace, not among production sources.
- Run relevant tests and the March migration regression before changing the frozen method.
- Extracted routing corpora under the external `asinfer/files` tree are read-only.
- Original RIBs in `/mnt/56DAEA7BDAEA56B1/pathInference/bgp` are optional archives, not training prerequisites.
