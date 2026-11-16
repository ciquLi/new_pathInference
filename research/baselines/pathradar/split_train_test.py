import random, os
from collections import defaultdict
import networkx as nx


fout_data = open('rib_data.txt', 'w')
fout_train = open('rib_train.txt', 'w')
fout_validation = open('rib_validation.txt', 'w')
fout_test = open('rib_test.txt', 'w')
fout_origintest = open('rib_origintest.txt', 'w')

vps, origins = defaultdict(set), defaultdict(set)
with open('done_rib.txt') as f:
    for line in f:
        path = line.strip().split('|')
        vps[path[0]].add(tuple(path))
        origins[path[-1]].add(tuple(path))

graph = nx.Graph()
with open('asrel_rib.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        [asn1, asn2, r] = line.strip().split('|')
        graph.add_edge(asn1, asn2)

choose_origins = set()
for key in origins:
    if graph.degree(key) <= 5:
        choose_origins.add(key)
    if len(choose_origins) == 100:
        break

train_path, validation_path = set(), set()
new_origins = defaultdict(set)
for vp in vps:
    randnum = random.randint(1, 10)
    if random.randint(1, 10) == 1:
        for path in vps[vp]:
            fout_test.write('|'.join(path) + '\n')
    else:
        for path in vps[vp]:
            if path[-1] in choose_origins:
                fout_origintest.write('|'.join(path) + '\n')
            else:
                fout_data.write('|'.join(path) + '\n')
                new_origins[path[-1]].add(path)
                if random.randint(1, 10) <= 5:
                    train_path.add(path)
                else:
                    validation_path.add(path)

transfer = set()
for key in new_origins:
    if random.randint(1, 20) == 1:
        for path in new_origins[key]:
            transfer.add(path)
train_path = train_path - transfer
validation_path = validation_path | transfer
for path in train_path:
    fout_train.write('|'.join(path) + '\n')
for path in validation_path:
    fout_validation.write('|'.join(path) + '\n')

fout_data.close()
fout_test.close()
fout_train.close()
fout_validation.close()
fout_origintest.close()

for cate in ['data', 'train']:
    linkfreq = defaultdict(int)
    linkorder = defaultdict(list)
    prev_frequency, next_frequency = defaultdict(int), defaultdict(int)
    
    graph = nx.Graph()
    with open('rib_' + cate + '.txt') as f:
        for line in f:
            path = line.strip().split('&')[0].split('|')[::-1]
            for i in range(len(path) - 1):
                graph.add_edge(path[i], path[i + 1])
                linkfreq[(path[i], path[i + 1])] += 1
                linkfreq[(path[i + 1], path[i])] += 1
            if len(path) > 1:
                for i in range(len(path)):
                    prev_frequency['|'.join(path[:i+1])] += 1
                    for j in range(i, len(path)):
                        next_frequency['|'.join(path[i:j+1])] += 1

    fout = open('linkfreq_' + cate + '.txt', 'w')
    for link in linkfreq:
        fout.write('|'.join(link) + ' ' + str(linkfreq[link]) + '\n')
    fout.close()

    fout = open('linkorder_' + cate + '.txt', 'w')
    for node in graph.nodes():
        freqorder = dict()
        for neighbor in graph.neighbors(node):
            freqorder[neighbor] = linkfreq[(node, neighbor)]
        freqorder = sorted(freqorder.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(5, len(freqorder))):
            (key, num) = freqorder[i]
            linkorder[node].append(key)
        fout.write(node + ' ' + '|'.join(linkorder[node]) + '\n')
                
    fout = open('joinfreq_' + cate + '.txt', 'w')
    for key in prev_frequency:
        fout.write('prev ' + key + ' ' + str(prev_frequency[key]) + '\n')
    for key in next_frequency:
        fout.write('next ' + key + ' ' + str(next_frequency[key]) + '\n')

    if cate == 'data':
        spread = defaultdict(set)
        with open('rib_' + cate + '.txt') as f:
            for line in f:
                path = line.strip().split('|')
                for i in range(len(path)):
                    for j in range(len(path)):
                        spread[path[i]].add(path[j])

        fout = open('core_shell.txt', 'w')
        for key in spread:
            fout.write(key + ' ' + str(len(spread[key])) + '\n')
        fout.close()