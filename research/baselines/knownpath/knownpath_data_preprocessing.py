#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:52:28 2019

@author: lixionglve
"""

import argparse
import os
import sqlite3
import subprocess
import pandas as pd
from multiprocessing import Pool
import ujson
import commonFunction as commonFun

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
FILES_PATH = Path(os.environ['KNOWNPATH_FILES_ROOT']).resolve()
files_path = str(FILES_PATH) + '/'


def get_dir(path, file_list, file_type, isChild=True):
    for file in os.listdir(path):
        file_path = path + '/' + file
        if os.path.isdir(file_path) and isChild:
            if file[0] != '.':
                get_dir(file_path, file_list, file_type)
        else:
            if os.path.splitext(file_path)[1] == file_type:
                father_path = os.path.dirname(file_path)
                date = os.path.split(father_path)[1]
                file_list.append([date, file_path])


def multi_process(function, argc, num_thread):
    p = Pool(num_thread)
    p.map(function, argc)
    p.close()
    p.join()


def knownpath_sure_path(parent, dataType=''):
    """Generate all suffix subpaths from raw CSVs, write to duplicated file (with duplicates)."""
    result_dir = files_path + 'data/aspath/path_by_vantage/' + parent
    as_dir = result_dir + '/path_without_prefix/%s' % (dataType)
    work_dir = Path(os.environ['KNOWNPATH_INDEX_ROOT']).resolve() / parent
    work_dir.mkdir(parents=True, exist_ok=True)
    all_file = str(work_dir / 'duplicated_knownpath_sure_paths.txt')
    if os.path.exists(all_file):
        os.remove(all_file)
    file_type = '.csv'
    file_list = []
    get_dir(as_dir, file_list, file_type)
    for row in file_list:
        file = row[1]
        if file.find('_paths.csv') > 0:
            continue
        print(row)
        df = pd.read_csv(file, names=['aspath'])
        tmp = []
        for v in df['aspath'].values:
            as_list = v.split('->')
            sure_paths = []
            l = len(as_list)
            if l < 3:
                sure_paths.append(v)
            else:
                for i in range(l - 1):
                    sure_paths.append('->'.join(as_list[i:]))
            tmp += sure_paths

        df = pd.DataFrame(list(set(tmp)))
        df.to_csv(all_file, index=False, header=None, mode='a')


def build_sqlite_db(parent, dataType=''):
    """Sort + uniq the duplicated sure paths, insert into SQLite with frequency, then clean up."""
    result_dir = files_path + 'data/aspath/path_by_vantage/' + parent
    as_dir = result_dir + '/path_without_prefix/%s' % dataType
    work_dir = Path(os.environ['KNOWNPATH_INDEX_ROOT']).resolve() / parent
    dup_file = str(work_dir / 'duplicated_knownpath_sure_paths.txt')
    db_path = str(work_dir / 'knownpath.db')

    if os.path.exists(db_path):
        raise FileExistsError('Existing index is preserved: ' + db_path)

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('''CREATE TABLE sure_paths (
        dest TEXT NOT NULL,
        src  TEXT NOT NULL,
        path TEXT NOT NULL,
        freq INTEGER NOT NULL
    )''')

    cmd = "sort '%s' | uniq -c" % dup_file
    batch = []
    batch_size = 100000
    total = 0

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        freq = int(parts[0])
        path_str = parts[1]
        as_list = path_str.split('->')
        if len(as_list) < 2:
            continue
        src = as_list[0]
        dest = as_list[-1]
        batch.append((dest, src, path_str, freq))
        total += 1

        if len(batch) >= batch_size:
            conn.executemany('INSERT INTO sure_paths VALUES (?,?,?,?)', batch)
            batch.clear()
            print('%d rows inserted' % total)

    proc.wait()
    if batch:
        conn.executemany('INSERT INTO sure_paths VALUES (?,?,?,?)', batch)

    print('%d total rows, creating index...' % total)
    conn.execute('CREATE INDEX idx_dest ON sure_paths(dest)')
    conn.commit()
    conn.close()

    os.remove(dup_file)
    print('SQLite DB built: %s (%d rows)' % (db_path, total))


def child(file, data_types=('test', 'train')):
    parent = file
    if parent == '.DS_Store':
        return

    for dataType in data_types:
        knownpath_sure_path(parent, dataType)
        build_sqlite_db(parent, dataType)


def main():
    file_list = ['20231001']
    for file in file_list:
        child(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, type=str)
    parser.add_argument("--data_type", choices=['train', 'test', 'all'], default='all')
    args = parser.parse_args()
    data_types = ('train',) if args.data_type == 'train' else ('test',) if args.data_type == 'test' else ('test', 'train')
    child(args.date, data_types)
