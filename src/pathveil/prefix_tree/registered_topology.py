"""Bind explicitly registered public snapshot dates to the frozen graph API."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

from pathveil.utils import read_json
from .public_prefix_tree import PublicPrefixTopology


class RegisteredPublicPrefixTopology(PublicPrefixTopology):
    def __init__(self, date, asrel, nodes, links, *, asrank_date):
        self.asrank_date = str(asrank_date)
        datetime.strptime(self.asrank_date, '%Y%m%d')
        if len(self.asrank_date) != 8 or self.asrank_date > str(date):
            raise ValueError('ASRank snapshot must be explicitly registered on or before the routing date')
        super().__init__(date, asrel, nodes, links)
        identity = dict(date=self.date, inputs_sha256=self.inputs_sha256, asrank_date=self.asrank_date)
        self.graph_identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_manifest(cls, manifest):
        data = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
        snapshot = data.get('auxiliary_dates', {}).get('asrank', data['date'])
        if snapshot == data['date']:
            return PublicPrefixTopology.from_manifest(data)
        files = data['public_files']
        return cls(data['date'], files['asrel'], files['asrank_nodes'], files['asrank_links'],
                   asrank_date=snapshot)

    def _check_date(self, row, counts):
        value = row.get('date')
        if not value:
            counts['undated_rows'] += 1
        elif str(value)[:10].replace('-', '') != self.asrank_date:
            raise ValueError('Public input row date differs from the registered snapshot: ' + str(value))

    def receipt(self):
        return dict(super().receipt(), registered_asrank_date=self.asrank_date)
