import os
import sqlite3
import networkx as nx
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
FILES_PATH = Path(os.environ['KNOWNPATH_FILES_ROOT']).resolve()
files_path = str(FILES_PATH) + '/'


class Env():
    def __init__(self, parent):
        self.parent = parent
        self.isGetData = False
        self.baseDir = str(FILES_PATH / 'data/aspath/path_by_vantage' / parent / 'RL_Graph') + '/'
        self.G = self.build_graph()
        self.surePaths = {}
        self.rel_dict = self.relationship_dict()
        self.dataType = 'train'
        self.maxLen = 20
        self.surePaths = {}
        self._db_conns = {}
        self._frequency_cache = None

    def _get_db(self, dataType):
        if dataType not in self._db_conns:
            override = os.environ.get('KNOWNPATH_TRAIN_DB') if dataType == 'train' else None
            db_path = override or (files_path + 'data/aspath/path_by_vantage/' + self.parent + '/path_without_prefix/%s/knownpath.db' % dataType)
            conn = sqlite3.connect(Path(db_path).as_uri() + '?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
            self._db_conns[dataType] = conn
        return self._db_conns[dataType]

    def build_graph(self):
        edge_path = self.baseDir + 'edge_frequency.json'
        fp = open(edge_path)
        ujson = __import__('ujson')
        str_js = ujson.load(fp)
        edge_dict = ujson.loads(str_js)
        fp.close()
        self.edge_freq_raw = edge_dict

        G = nx.Graph()
        for key in edge_dict.keys():
            edge = key.split('-')
            u = edge[0]
            v = edge[1]
            G.add_edge(u, v)
            edge = key.split('-')
            u = edge[0]
            v = edge[1]
            G.add_edge(u, v)
        degrees = list(G.degree())
        deg = 0
        for degree in degrees:
            d = degree[1]
            deg += d
        avgDegree = deg / len(degrees)
        print('date:%s,numNodes:%d,numEdges:%d,avgDegree%.2f' % (self.parent, len(list(G.nodes)), len(list(G.edges)), avgDegree))
        print('2m/n:', 2 * len(list(G.edges)) / len(list(G.nodes)))
        return G

    @property
    def frequency_dict(self):
        if self._frequency_cache is None:
            conn = self._get_db('train')
            rows = conn.execute('SELECT path, freq FROM sure_paths').fetchall()
            self._frequency_cache = {row['path']: row['freq'] for row in rows}
            print('frequency_dict loaded: %d entries' % len(self._frequency_cache))
        return self._frequency_cache

    def getSurePaths_by_dest(self, dataType, destASN):
        conn = self._get_db(dataType)
        rows = conn.execute('SELECT src, path FROM sure_paths WHERE dest = ?', (destASN,)).fetchall()
        surePaths = {}
        for row in rows:
            src = row['src']
            as_list = row['path'].split('->')
            if src in surePaths:
                surePaths[src].append(as_list)
            else:
                surePaths[src] = [as_list]
        self.surePaths = surePaths
        return surePaths

    def getBaseASset(self, dataType, destASN):
        conn = self._get_db(dataType)
        rows = conn.execute('SELECT DISTINCT src FROM sure_paths WHERE dest = ?', (destASN,)).fetchall()
        baseASDict = {row['src']: 1 for row in rows}
        return baseASDict

    def get_exist_path_by_ASN(self, VASN, dataType):
        file = files_path + 'data/aspath/path_by_vantage/' \
             + self.parent + '/path_without_prefix/%s/' % (dataType) + VASN + '.csv'
        df = pd.read_csv(file, names=['aspath'])
        src_dst_path = {}
        for row in df['aspath'].values:
            aspath = row
            as_list = aspath.split('->')
            dst = as_list[-1]
            key = VASN + '-' + dst
            try:
                tmp = src_dst_path[key]
                tmp.append(as_list)
                src_dst_path[key] = tmp
            except:
                src_dst_path[key] = [as_list]
        return src_dst_path

    def relationship_dict(self):
        as_relationship_file = files_path + 'data/aspath/%s.as-rel.txt' % (self.parent)
        df = pd.read_csv(as_relationship_file, names=['rel'])
        rel = df['rel'].tolist()
        d = {}
        for r in rel:
            as1 = r.split('|')[0]
            as2 = r.split('|')[1]
            relationship = int(r.split('|')[2])
            try:
                tmp = d[as1]
                tmp[as2] = relationship
                d[as1] = tmp
            except:
                d1 = {}
                d1[as2] = relationship
                d[as1] = d1
        return d

    def getMaxLen(self, existPath):
        maxLen = 0
        for key in existPath.keys():
            paths = existPath[key]
            for path in paths:
                l = len(path)
                if l > maxLen:
                    maxLen = l
        print('maxLen:%d' % (maxLen))
        self.maxLen = maxLen
        return maxLen
