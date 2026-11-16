# Local Reproduction Notes

This repository can be reproduced against the local MetaGMPI dataset under:

- `/media/lxl/data/asinfer/files/data/aspath/bgp/20251201`
- `/media/lxl/data/asinfer/files/data/aspath/path_by_vantage/20251201`
- `/media/lxl/data/asinfer/files/data/aspath/20251201.as-rel.txt`

## 1. Build raw PathRadar input

```bash
python tools/prepare_local_aspaths.py --date 20251201
cp /media/lxl/data/asinfer/files/data/aspath/20251201.as-rel.txt asrel.txt
```

This creates:

- `aspaths_20251201_raw.txt`
- `asrel.txt`

## 2. Download the matching PeeringDB snapshot

PathRadar's `unique_path.py` uses PeeringDB to remove route-server ASNs. Put the snapshot in this repo, for example:

- `peeringdb_2025_12.json`

Then run:

```bash
python unique_path.py -i=aspaths_20251201_raw.txt -p=peeringdb_2025_12.json
mv aspaths.txt aspaths_20251201_unique.txt
ln -sf aspaths_20251201_unique.txt aspaths.txt
```

## 3. Build valley-free paths

```bash
python clean_alldata.py
```

This creates:

- `done_rib.txt`
- `asrel_rib.txt`

## 4. Align the split with MetaGMPI

Instead of using the repository's default random split, reuse MetaGMPI's existing
`path_without_prefix/train` and `path_without_prefix/test` directories:

```bash
python tools/split_train_test_metagmpi.py --date 20251201
```

This creates:

- `rib_data.txt`
- `rib_train.txt`
- `rib_validation.txt`
- `rib_test.txt`
- `rib_origintest.txt`
- `linkfreq_data.txt`
- `linkorder_data.txt`
- `joinfreq_data.txt`
- `linkfreq_train.txt`
- `linkorder_train.txt`
- `joinfreq_train.txt`
- `core_shell.txt`

## 5. Continue with the native PathRadar pipeline

```bash
python combine_path.py
python gen_false_sample.py -m 10
python pathradar_train.py -r 0 -q 1
python pathradar_prediction.py -r 0 -t 0.5 -m train -q 1
```

After round 0 is stable, continue with round 1 and round 2.

## Notes

- The local `20251201` dataset is large. Expect the raw conversion step to take several minutes.
- The fair comparison setup should reuse MetaGMPI's existing train/test split instead of PathRadar's random split.
- If you later want strict paper-style reproduction, switch back to `split_train_test.py` and treat that as a separate experiment from the fair-comparison run.
