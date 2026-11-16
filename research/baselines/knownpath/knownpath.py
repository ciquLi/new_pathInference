import environment
import commonFunction as commonFun
import get_evaluate_data as ged
import numpy as np
import pandas as pd
import time
import os
import ujson

try:
    from cpp_engine import KnownpathEngine
    _CPP_AVAILABLE = True
except Exception:
    _CPP_AVAILABLE = False
""" 
path: list
"""
def get_path_ulen(env,path):
    ulen = 0
    curSurePath = ''
    for i in range(len(path)-1):
        key = path[i]
        try:
            surePaths = env.surePaths[key]
            for spath in surePaths:
                if spath == path[i:]:
                    ulen = len(path) - len(path[i:])
                    curSurePath = '->'.join(path[i:])
                    break
            if curSurePath != '':
                break
        except:
            continue
    return ulen,curSurePath


""" 
p1,p2: path, list

"""
def _key_path(env, p):
    ulen, sure_path = get_path_ulen(env, p)
    freq = env.frequency_dict.get(sure_path, 1)
    return (len(p), ulen, -freq, int(p[1]) if len(p) > 1 else 0)

def sort_paths_SPF(env, paths):
    paths.sort(key=lambda p: _key_path(env, p))
    return paths



"""  """
def initActiveQueue(env,dest):
    rib_in = env.getSurePaths_by_dest('train',dest)    #所有目的地为dest的paths
    baseASDict = env.getBaseASset('train',dest)
    baseASset = list(baseASDict.keys())
    queue = {}
    for v in baseASset:
        queue[v] = 1
        paths = rib_in[v]    #list of list [[as1,as2,...,dest],[as1,as3,...,dest],...]
        paths = sort_paths_SPF(env,paths)
        rib_in[v] = paths
    return queue,baseASDict,rib_in    
        

def knownPath(env, dest):
    queue, baseASDict, rib_in = initActiveQueue(env, dest)
    while queue:
        u = next(iter(queue))
        queue.pop(u)
        p_u = rib_in[u][0]
        for v in env.G.neighbors(u):
            if v in baseASDict:
                continue
            v_p_u = [v] + p_u
            if len(v_p_u) > len(set(v_p_u)):
                continue
            if commonFun.valley_free(v_p_u, env.rel_dict):
                try:
                    tmppath = rib_in[v][0].copy()
                    for path in rib_in[v]:
                        if path[1] == u:
                            path = v_p_u
                            break
                except KeyError:
                    tmppath = []
                    rib_in[v] = [v_p_u]
                paths = sort_paths_SPF(env, rib_in[v])
                rib_in[v] = paths
                if tmppath != rib_in[v][0]:
                    if v not in queue:
                        queue[v] = 1

    return rib_in
                    
  
                   
def analyze2(env,dest,srcs,rib_in):
    evaluate_paths = env.ground_truth[dest]
    # print(evaluate_paths)
    cnt = 0
    result = []
    detail_result = []
    exact_result = []
    srcs_dict = {}
    for src in srcs:
        srcs_dict[src] = 1
    for key in evaluate_paths.keys():
        try:
            srcs_dict[key]
        except:
            continue
        # if key not in srcs:
        #     continue
        paths = evaluate_paths[key]
        try:
            inferred_path = rib_in[key][0]
        except:
            inferred_path = []
        
        maximum = 0
        similarites = []
        # sort the paths in paths by length
        paths = sorted(paths,key=lambda x:len(x))
        # paths = paths[:1]
        for path in paths:
            if len(path) == 0:
                continue
            try:
                similarity = commonFun.jaccard_index(inferred_path,path)  
            except:
                similarity = 0 
            similarites.append(similarity)
            if similarity > maximum:
                maximum = similarity
        mean_similarity = np.mean(similarites)
        detail_result.append([dest,key,mean_similarity,maximum])
        if len(inferred_path) > 0 and similarity != 0:
            cnt += 1
        result.append(maximum)
        if maximum >= 1.0:
            exact_result.append(maximum)

    # fp = open('result.txt','a')
    txt = 'dest:%s,src_num:%d,result_len:%d'%(dest,len(srcs),len(result))
    # print(txt)
    # fp.write(txt+'\n')
    txt += ',evaluate_path_num:%d'%(len(srcs))
    # print(txt)
    # fp.write(txt+'\n')
    txt +=',avg accuracy:%.2f'%(sum(result)/len(srcs))
    # print(txt)
    # fp.write(txt+'\n')
    txt += ',avg esr:%.2f'%(sum(exact_result)/len(srcs))
    print(txt)
    # fp.write(txt+'\n')
    # fp.close()
    results = [dest,len(srcs),cnt,sum(result),sum(exact_result)]
    return results, detail_result

def inferpath(env, dest_srcs, cache_dir, cpp_engine=None):
    t1 = time.time()
    dest = dest_srcs[1]
    filename = cache_dir / (dest + '.json')
    if filename.exists():
        with open(filename) as fp:
            rib_in = ujson.load(fp)
        print('load successfully')
    else:
        try:
            if cpp_engine:
                sure_paths = env.getSurePaths_by_dest('train', dest)
                base_as = env.getBaseASset('train', dest)
                rib_in = cpp_engine.compute(sure_paths, base_as)
            else:
                rib_in = knownPath(env, dest)
        except Exception:
            rib_in = {}
    print('over')
    srcs = dest_srcs[2]
    t2 = time.time()
    try:
        res,detail_result = analyze2(env,dest,srcs,rib_in)
    except:
        res = [0,0,0,0,0]
        detail_result = []
    res.append(t2-t1)
    return res,detail_result

def select_intensive_vases(env):
    measured_vantage_dst = commonFun.get_measured_vantage_dst(env.baseDir)
    train_sparse_vas_dests, train_intensive_vas_dests = commonFun.classify_vas(measured_vantage_dst,env.border)    #train里面的vas-dests分类情况
    unmeasure_vas_dests = commonFun.get_unmeasured_vantage_dst(env.baseDir,'test')
    test_sparse_vas_dests, test_intensive_vas_dests, test_unSeen = commonFun.get_classified_vas(unmeasure_vas_dests,env.G,list(train_sparse_vas_dests.keys()), list(train_intensive_vas_dests.keys()))    #test里面的分类情况
    intensive_asns = list(test_intensive_vas_dests.keys())

    # asn_avglen = env.get_vas_pathAvgLen()
    # asn_collected_path_avglen = {}
    # for asn,avglen,num in asn_avglen:
    #     asn_collected_path_avglen[asn] = avglen
        
    # intensive_asns.sort(key=lambda x:asn_collected_path_avglen[x])
    vases = intensive_asns
    return vases

def knownpath(env, all_dest_srcs, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = result_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    result_file = str(result_dir) + '/%s-%s-%s-result.csv' % (env.curDate, env.cur_node_idx, env.as_type)
    detail_file = str(result_dir) + '/%s-%s-%s-detail-result.csv' % (env.curDate, env.cur_node_idx, env.as_type)
    done_dests = set()
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dest = line.split(',')[0]
                done_dests.add(dest)
        print('Skipping %d already-processed dests' % len(done_dests))

    engine = None
    if _CPP_AVAILABLE:
        engine = KnownpathEngine(env)
        print('Using C++ engine')

    for dest_srcs in all_dest_srcs:
        dest = dest_srcs[1]
        if dest in done_dests:
            print('skip %s (already done)' % dest)
            continue
        result, detail_result = inferpath(env, dest_srcs, cache_dir, engine)
        df = pd.DataFrame([result])
        df.to_csv(result_file, index=False, header=None, mode='a')
        df_detail = pd.DataFrame(detail_result)
        df_detail.to_csv(detail_file, index=False, header=None, mode='a')

    if engine:
        engine.close()


def split_aslit(alist,n):
    list_of_vases = []
    part = len(alist)//n
    for i in range(n):
        if i == n-1:
            last_part = alist[i*part:]
            if len(last_part) > part:
                list_of_vases.append(last_part[:part])
                for idx,item in enumerate(last_part[part:]):
                    list_of_vases[idx].append(item)
            else:
                list_of_vases.append(last_part)
        else:
            list_of_vases.append(alist[i*part:(i+1)*part])
    return list_of_vases

import argparse
from pathlib import Path

PROJECT_DIR = Path(os.environ['KNOWNPATH_RUN_ROOT']).resolve()
FILES_PATH = Path(os.environ['KNOWNPATH_FILES_ROOT']).resolve()


def ground_truth(parent,dests):
    result_dir = FILES_PATH / 'data/aspath/path_by_vantage' / parent
    as_dir = result_dir / 'path_without_prefix' / 'test'
    grount_truth_by_dest = {}
    files = os.listdir(as_dir)
    dests_dict = {}
    for dest in dests:
        dests_dict[dest] = 1
    for file in files:
        if file[-4:] != '.csv':
            continue
        file = as_dir / file
        df = pd.read_csv(file, names=['aspath'])
        for v in df['aspath'].values:
            as_list = v.split('->')
            src = as_list[0]
            dst = as_list[-1]
            if dst in dests_dict:
                if dst in grount_truth_by_dest:
                    if src not in grount_truth_by_dest[dst]:
                        grount_truth_by_dest[dst][src] = [as_list]
                    else:
                        tmp_list = grount_truth_by_dest[dst][src]
                        tmp_list.append(as_list)
                        grount_truth_by_dest[dst][src] = tmp_list
                else:
                    grount_truth_by_dest[dst] = {src:[as_list]}
                      
    return grount_truth_by_dest       

def select_test_vases(date, num_vases):
    test_dir = FILES_PATH / 'data/aspath/path_by_vantage' / date / 'path_without_prefix' / 'test'
    metadata_path = FILES_PATH / 'data/aspath/path_by_vantage' / date / 'RL_Graph' / 'vantage_dst_test.json'
    with open(metadata_path) as fp:
        test_vas_dests = ujson.loads(ujson.load(fp))

    files = [path for path in test_dir.glob('*.csv') if path.stem in test_vas_dests]
    source_filter = os.environ.get('KNOWNPATH_SOURCES')
    if source_filter:
        selected = set(source_filter.split(','))
        files = [path for path in files if path.stem in selected]
    files.sort(key=lambda path: path.stat().st_size, reverse=True)
    if len(files) < num_vases:
        raise ValueError('Requested %d test VAs, but only found %d.' % (num_vases, len(files)))
    return [path.stem for path in files[:num_vases]]

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_num", default=1, type=int)    # split the tasks
    parser.add_argument("--cur_idx", default=0, type=int)
    parser.add_argument("--cur_node_idx", default=0, type=int)
    parser.add_argument("--date", required=True, type=str)
    parser.add_argument("--as_type", default='knownpath', type=str)
    parser.add_argument("--num_vas", default=10, type=int)
    parser.add_argument("--max_dests", default=None, type=int)
    parser.add_argument("--range_start", default=None, type=int)
    parser.add_argument("--range_end", default=None, type=int)
    args = parser.parse_args()
    split_num = args.split_num
    cur_idx = args.cur_idx
    cur_node_idx = args.cur_node_idx
    as_type = args.as_type
    num_vases = args.num_vas
    date = args.date
    result_dir = PROJECT_DIR / 'results' / date
    # dates = []
    # for date in dates:
    env = environment.Env(date)
    env.border = 7000
    env.cur_node_idx = cur_node_idx
    env.as_type = as_type
    vases = select_test_vases(date, num_vases)
    dest_srcs = ged.get_evaluate_data(env,vases)
    if args.max_dests is not None:
        dest_srcs = dest_srcs[-args.max_dests:]
    print('len:', len(dest_srcs))
    total_aps = 0
    for row in dest_srcs:
        total_aps += len(row[2])
    print('total_aps:', total_aps)
    splitted_list = split_aslit(dest_srcs,split_num)
    
    cur_dest_srcs = splitted_list[cur_idx]
    if args.range_start is not None or args.range_end is not None:
        cur_dest_srcs = cur_dest_srcs[args.range_start:args.range_end]
    print('curlen:',len(cur_dest_srcs))
    cur_aps = 0
    dests = []
    for row in cur_dest_srcs:
        cur_aps += len(row[2])
        dests.append(row[1])
    print('cur_aps:', cur_aps)
    # env = environment.Env(date)
    # env.border = 7000
    env.cur_node_idx = cur_node_idx
    env.ground_truth = ground_truth(date,dests)
    env.curDate = date
    print('len ground truth:', len(env.ground_truth))
    knownpath(env, cur_dest_srcs, result_dir)
        
        

    
    
    
    
    

    
    
    

    
