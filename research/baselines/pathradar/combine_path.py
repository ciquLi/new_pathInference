from collections import defaultdict
import os
from pathlib import Path

REPO_DIR = Path(os.environ.get('PATHRADAR_RUN_DIR', Path(__file__).resolve().parent.parent))
INTERMEDIATE_DIR = REPO_DIR / 'intermediate'

for cate in ['done_rib', 'rib_train', 'rib_validation', 'rib_data', 'rib_test', 'rib_origintest']:
	name = cate.split('_')[1]
	target_dir = INTERMEDIATE_DIR / name
	if not target_dir.exists():
		target_dir.mkdir(parents=True, exist_ok=True)

	info = defaultdict(set)
	with open(INTERMEDIATE_DIR / f'{cate}.txt') as f:
		for line in f:
			path = line.strip().split('|')
			info[path[-1]].add('|'.join(path))

	for key in info:
		fout = open(target_dir / key, 'w')
		all_path = set()
		for path in info[key]:
			ases = path.split('|')
			for i in range(len(ases) - 1):
				all_path.add('|'.join(ases[i:]))
		fout.write(' '.join(all_path) + '\n')
		fout.close()
