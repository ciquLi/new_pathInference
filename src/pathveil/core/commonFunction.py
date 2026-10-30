"""Shared routing and query utilities used by the production PG workflow."""
import os
import pandas as pd
import shutil
import random
from pathlib import Path
import ujson
from pathveil.core.io import open_data as open, read_exists, PandasReadProxy

files_path = os.path.join(
    os.environ.get("ASINFER_FILES_ROOT", "/run/media/lxl/data/asinfer/files"), ""
)


def load_json(filepath):
    """从文件加载JSON数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        return ujson.load(f)


def split_list_evenly(items, n):
    if n <= 0:
        raise ValueError("n must be positive")
    groups = []
    part = len(items) // n
    for i in range(n):
        if i == n - 1:
            last_part = items[i * part :]
            if len(last_part) > part:
                groups.append(last_part[:part])
                for idx, item in enumerate(last_part[part:]):
                    groups[idx].append(item)
            else:
                groups.append(last_part)
        else:
            groups.append(items[i * part : (i + 1) * part])
    return groups


def get_known_as_relationship(relationship, as1, as2):
    """Return an explicitly recorded relation; missing evidence is None."""
    (as1, as2) = (str(as1), str(as2))
    value = relationship.get(as1, {}).get(as2)
    if value in (-1, 0, 1, 2):
        return value
    value = relationship.get(as2, {}).get(as1)
    if value in (-1, 0, 1):
        return -value
    return 2 if value == 2 else None


def get_as_relationship(relationship, as1, as2):
    try:
        rel = relationship[as1][as2]
    except:
        try:
            rel = -relationship[as2][as1]
        except:
            rel = 2
    return rel


def valley_free(as_list, rel_dict):
    res = True
    relationship = []
    for i in range(len(as_list) - 1):
        rel = get_as_relationship(rel_dict, as_list[i], as_list[i + 1])
        relationship.append(rel)
    if relationship.count(0) > 1:
        res = False
        return res
    if -1 in relationship:
        pos = relationship.index(-1)
        if 1 in relationship[pos + 1 :]:
            res = False
            return res
    if -1 in relationship:
        pos = relationship.index(-1)
        if 0 in relationship[pos + 1 :]:
            res = False
            return res
    if 0 in relationship:
        pos = relationship.index(0)
        if 1 in relationship[pos + 1 :]:
            res = False
            return False
    return res


def jaccard_index(li1, li2):
    set1 = set(li1)
    set2 = set(li2)
    if not set1 and (not set2):
        return 0
    return len(set1 & set2) / len(set1 | set2)


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def get_measured_vantage_dst(baseDir):
    vantage_dst_path = baseDir + "vantage_dst.json"
    fp = open(vantage_dst_path)
    str_js = ujson.load(fp)
    vantage_dst = ujson.loads(str_js)
    fp.close()
    return vantage_dst


def get_unmeasured_vantage_dst(baseDir, dataType):
    vantage_dst_path = baseDir + "vantage_dst_%s.json" % dataType
    fp = open(vantage_dst_path)
    str_js = ujson.load(fp)
    vantage_dst = ujson.loads(str_js)
    fp.close()
    return vantage_dst


def select_meta_vases(date, meta_test_as_num, target_vases=None):
    if target_vases:
        if isinstance(target_vases, str):
            target_vases = [item.strip() for item in target_vases.split(",") if item.strip()]
        return [str(vas) for vas in target_vases]
    file_dir = files_path + "data/aspath/path_by_vantage/" + date + "/path_without_prefix"
    file_type = ".csv"
    file_list = []
    for file in os.listdir(file_dir):
        if file.endswith(file_type):
            file_list.append(file_dir + "/" + file)
    file_list.sort(key=lambda x: os.path.getsize(x), reverse=True)
    meta_files = file_list[:meta_test_as_num]
    meta_vases = []
    for file in meta_files:
        vas = file.split("/")[-1].split(".")[0]
        meta_vases.append(vas)
    return meta_vases


pd = PandasReadProxy(pd)
