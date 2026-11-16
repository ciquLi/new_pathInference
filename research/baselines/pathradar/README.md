# PathRadar reproduction

This is an isolated copy of the PathRadar implementation under the sibling
`asinfer/pathRadar` directory. It uses the registered BGP dates and their
train/test split. Training intermediates are taken only from train paths;
test CSVs are used for final evaluation.

For 20260201, the existing intermediate SQLite and auxiliary files were
verified against the current train/test CSVs and are opened immutable and
read-only. The old 20260301 and 20260401 staging omitted `34927.csv` from
train, so those dates are rebuilt from the current input, as are 20260501
and 20260601. Trained models and full-query evaluation outputs are written
under `runs/baselines/pathradar/YYYYMMDD/`.

```bash
python research/baselines/pathradar/run_baseline.py 20260201
```

The batch dispatcher waits for all five KnownPath dates to complete, then
runs PathRadar one date at a time. It trains rounds 0, 1, and 2. Each of the
first two rounds predicts over all training destinations to supply negatives
for the next round; round 2 is evaluated on every registered test pair.
Completed round-0 models and results are reused. After round-2 evaluation
passes, local intermediate data and training predictions are removed, while
models, result JSON, identity records, and logs are retained. The 20261015
traceroute date is excluded.
