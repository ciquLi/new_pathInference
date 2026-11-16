
import os

#获取后缀为file_type文件的路径
def get_dir(path,file_list,file_type):  
    for file in os.listdir(path):  
        file_path = path + '/' +file
        if os.path.isdir(file_path):
            if file[0] != '.':
                get_dir(file_path,file_list,file_type)
        else:
            if os.path.splitext(file_path)[1]==file_type:
                #获取当前文件所在目录名以及当前文件的相对路径
                father_path = os.path.dirname(file_path)
                date = os.path.split(father_path)[1]
                file_list.append([date,file_path])



def get_as_relationship(relationship,as1,as2):
    #<provider-as>|<customer-as>|-1 反过来为1 peer peer 为0  ,sibling - sibling 2
    try:
        rel = relationship[as1][as2]
    except:
        try:
            rel = -relationship[as2][as1]
        except:
            rel = 2
    
    return rel

############
'''
判断AS路径是否符合valley-free原则
准则一:一条 AS 路径中最多只有一条 P2P 连接;
准则二:一条 AS 路径中，如果有一条 P2C 连接，它后面就不可能是 C2P 连接 ，只可能是 P2C 或 S2S 连接;
准则三:P2C 连接后不可能是 P2P 连接;
准则四:P2P 连接后不可能是 C2P 连接，只可能是 P2C 或 S2S 连接。

as_list:  [as1,as2,as3,as4]
'''
############
def valley_free(as_list,rel_dict):
    res = True
    relationship = []
    for i in range(len(as_list)-1):
        rel = get_as_relationship(rel_dict,as_list[i],as_list[i+1])
        relationship.append(rel)
    if relationship.count(0) > 1:
        res = False
        return res
    if -1 in relationship:
        pos = relationship.index(-1)
        if 1 in relationship[pos+1:]:
            res = False
            return res
    if -1 in relationship:
        pos = relationship.index(-1)
        if 0 in relationship[pos+1:]:
            res = False
            return res
    if 0 in relationship:
        pos = relationship.index(0)
        if 1 in relationship[pos+1:]:
            res = False
            return False
    return res

#jaccard相似性,输入为两个LIST
def jaccard_index(li1,li2):
    intersection = []
    union = li2.copy()
    for l in li1:
        if l in li2:
            intersection.append(l)
        else:
            union.append(l)
    similarity = len(intersection)/len(union)
    return similarity

import shutil
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
            
def get_min_length_idx(paths):
    min_epath_length = 100    #随便设置的一个路径长度不会超过的数字
    min_idx = 0
    for idx,epath in enumerate(paths):
        if len(epath) < min_epath_length:
            min_epath_length = len(epath)
            min_idx = idx
    return min_idx



import random
 

def data_split(full_list, ratio, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2
 
 
""" 将一个dict按key排序，并输出list """
def sort_dict(length_dict):
    keys = list(length_dict.keys())
    keys.sort()
    length_list = []
    for key in keys:
        length_list.append([key,length_dict[key]])
    return length_list

# kpaths: list of list
def get_shortest_from_kpaths(kpaths):
    sort_by_length = []
    for path in kpaths:
        sort_by_length.append([len(path), path])
    sort_by_length.sort()
    shortest = sort_by_length[0][1]  #取出最短一个的长度
    return shortest

""" 获得前n个的索引 n必须小于alist的长度"""
def get_top_n_index(alist,n):
    sorted_list = alist.copy()
    sorted_list.sort(reverse=True)
    top_n = sorted_list[:n]
    idxs = []
    for top in top_n:
        idxs.append(alist.index(top))
    return idxs

from pathlib import Path
def getFileInDir(file_path,fileType):
    p = Path(file_path)
    assert(p.is_dir())
    files = sorted(p.glob('*.%s'%(fileType)))
    if len(files) < 1:
        raise RuntimeError('No %s datasets found'%(fileType))
    return files

import ujson
def get_measured_vantage_dst(baseDir):
    vantage_dst_path = baseDir + 'vantage_dst.json'
    fp = open(vantage_dst_path)
    str_js = ujson.load(fp)
    vantage_dst = ujson.loads(str_js)
    fp.close()
    return vantage_dst

def get_unmeasured_vantage_dst(baseDir,dataType):
    vantage_dst_path = baseDir + 'vantage_dst_%s.json'%(dataType)
    fp = open(vantage_dst_path)
    str_js = ujson.load(fp)
    vantage_dst = ujson.loads(str_js)
    fp.close()
    return vantage_dst

#判断train 中vas是sparse还是intensive
def classify_vas(vantage_dst,border):
    sparse, intensive = {}, {}
    for vas in vantage_dst.keys():
        dests = vantage_dst[vas]
        num = len(dests)
        if num < border:
            sparse[vas] = dests
        else:
            intensive[vas] = dests
    return sparse, intensive

#从unmeasured APs里面找出path-sparse vas, path-intensive vas, unseen vas
def get_classified_vas(unmeasure_vas_dests,G,sparse,intensive):
    sparse_vas_dests = {}
    intensive_vas_dests = {}
    unSeen = {}
    for vas in unmeasure_vas_dests:
        if vas in intensive:
            dests = unmeasure_vas_dests[vas]
            inGraph = []
            notInGraph = []
            for dest in dests:
                if dest in G:
                    inGraph.append(dest)
                else:
                    notInGraph.append(dest)
            if len(inGraph) > 0:
                intensive_vas_dests[vas] = inGraph
            try:
                tmp = unSeen[vas]
                tmp += notInGraph
                unSeen[vas] = tmp
            except:
                if len(notInGraph) > 0:
                    unSeen[vas] = notInGraph
        elif vas in sparse:
            dests = unmeasure_vas_dests[vas]
            inGraph = []
            notInGraph = []
            for dest in dests:
                if dest in G:
                    inGraph.append(dest)
                else:
                    notInGraph.append(dest)
            if len(inGraph) > 0:
                sparse_vas_dests[vas] = inGraph
            try:
                tmp = unSeen[vas]
                tmp += notInGraph
                unSeen[vas] = tmp
            except:
                if len(notInGraph) > 0:
                    unSeen[vas] = notInGraph
        else:
            if vas not in G:
                notInGraph = unmeasure_vas_dests[vas]
            else:
                dests = unmeasure_vas_dests[vas]
                inGraph = []
                notInGraph = []
                for dest in dests:
                    if dest in G:
                        inGraph.append(dest)
                    else:
                        notInGraph.append(dest)
                if len(inGraph) > 0:
                    sparse_vas_dests[vas] = inGraph
            try:
                tmp = unSeen[vas]
                tmp += notInGraph
                unSeen[vas] = tmp
            except:
                if len(notInGraph) > 0:
                    unSeen[vas] = notInGraph
    return sparse_vas_dests, intensive_vas_dests,unSeen