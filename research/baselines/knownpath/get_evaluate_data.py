import ujson

import environment
#判断train 中vas是sparse还是intensive
def classify_vas(env,border):
    vantage_dst = get_measured_vantage_dst(env)
    sparse, intensive = {}, {}
    for vas in vantage_dst.keys():
        dests = vantage_dst[vas]
        num = len(dests)
        if num < border:
            sparse[vas] = dests
        else:
            intensive[vas] = dests
    return sparse, intensive

def get_measured_vantage_dst(env):
    vantage_dst_path = env.baseDir + 'vantage_dst.json'
    fp = open(vantage_dst_path)
    str_js = ujson.load(fp)
    vantage_dst = ujson.loads(str_js)
    fp.close()
    return vantage_dst

def get_unmeasured_vantage_dst(env,dataType):
    vantage_dst_path = env.baseDir + 'vantage_dst_%s.json'%(dataType)
    fp = open(vantage_dst_path)
    str_js = ujson.load(fp)
    vantage_dst = ujson.loads(str_js)
    fp.close()
    return vantage_dst

#从unmeasured APs里面找出path-sparse vas, path-intensive vas, unseen vas
def get_classified_vas(env,sparse,intensive,dataType):
    unmeasure_vas_dests = get_unmeasured_vantage_dst(env,dataType)
    sparse_vas_dests = {}
    intensive_vas_dests = {}
    unSeen = {}
    for vas in unmeasure_vas_dests:
        if vas in intensive:
            dests = unmeasure_vas_dests[vas]
            inGraph = []
            notInGraph = []
            for dest in dests:
                if dest in env.G:
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
                if dest in env.G:
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
            if vas not in env.G:
                notInGraph = unmeasure_vas_dests[vas]
            else:
                dests = unmeasure_vas_dests[vas]
                inGraph = []
                notInGraph = []
                for dest in dests:
                    if dest in env.G:
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



def get_intensive_vas_dests(vases,train_intensive_vas_dests,test_intensive_vas_dests):
    train_insensive = {}
    test_intensive = {}
    for vas in vases:
        train_dests = train_intensive_vas_dests[vas]
        test_dests = test_intensive_vas_dests[vas]
        train_insensive[vas] = train_dests
        test_intensive[vas] = test_dests
    return train_insensive, test_intensive
    
    
def get_evaluate_data(env,vases):
    
    # border = env.border
    # train_sparse_vas_dests, train_intensive_vas_dests = classify_vas(env,border)    #train里面的vas-dests分类情况
    # test_sparse_vas_dests, test_intensive_vas_dests, test_unSeen = \
    # get_classified_vas(env,list(train_sparse_vas_dests.keys()), list(train_intensive_vas_dests.keys()),'test')    #test里面的分类情况

    # train_insensive, test_intensive = \
    # get_intensive_vas_dests(vases,train_intensive_vas_dests,test_intensive_vas_dests)
    
    test_vas_dests = get_unmeasured_vantage_dst(env,'test')
    meta_vas_dests = {}
    for vas in vases:
        meta_vas_dests[vas] = test_vas_dests[vas]
    

    dest_srcs_dict = convert(meta_vas_dests)
    print('Evaluate Data Obtained!')
    return dest_srcs_dict

def convert(src_dests_dict):
    dest_srcs_dict = {}
    for src in src_dests_dict.keys():
        dests = src_dests_dict[src]
        for dest in dests:
            try:
                tmp = dest_srcs_dict[dest]
                tmp.append(src)
                dest_srcs_dict[dest] = tmp
            except:
                dest_srcs_dict[dest] = [src]
    dest_srcs = []

    for key in dest_srcs_dict.keys():
        items = dest_srcs_dict[key]
        items = list(set(items))
        dest_srcs.append([len(items),key,items])
        # dest_srcs_dict[key] = items
    dest_srcs.sort(reverse=True)
    # for row in dest_srcs:
    #     key = row[1]
    #     items = row[2]
    #     dest_srcs_dict[key] = items
    return dest_srcs



        