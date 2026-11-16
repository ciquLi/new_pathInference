# KnownPath reproduction

This is an isolated copy of the legacy KnownPath C++ implementation. It
reads BGP-derived train indexes and test queries from the dated `files`
snapshot. The inference engine reads only the train index; held-out test
paths are loaded for scoring after inference. No external corpus is
modified.

Run one BGP date with:

```bash
python research/baselines/knownpath/run_baseline.py 20260201 --workers 4
```

Results and identity manifests are written under
`runs/baselines/knownpath/YYYYMMDD/`. For 20260501 and 20260601, train
indexes are built locally under `runs/baselines/knownpath/indexes/` with:

```bash
KNOWNPATH_FILES_ROOT=/run/media/lxl/data/asinfer/files \
KNOWNPATH_INDEX_ROOT="$PWD/runs/baselines/knownpath/indexes" \
python research/baselines/knownpath/knownpath_data_preprocessing.py \
  --date 20260501 --data_type train
```

The 20261015 traceroute date has a different evaluation format and is
excluded from this BGP runner. This experiment uses every pair for each
configured test source, not a 10,000-pair evaluation sample. The 20260201
run began with two workers and was repartitioned into four after preserving
the completed destinations; `IDENTITY_INITIAL.json` records the first
dispatch.
