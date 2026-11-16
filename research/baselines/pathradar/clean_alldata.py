from collections import defaultdict
import networkx as nx
import copy
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_DIR / 'raw_data'
INTERMEDIATE_DIR = REPO_DIR / 'intermediate'

provider = defaultdict(set)
peer = defaultdict(set)
customer = defaultdict(set)
relationship = dict()

with open(RAW_DIR / 'asrel.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        [asn1, asn2, r] = line.strip().split('|')
        relationship[(asn1, asn2)] = r
        if r == '0':
            peer[asn1].add(asn2)
            peer[asn2].add(asn1)
        else:
            customer[asn1].add(asn2)
            provider[asn2].add(asn1)

def get_rel(path):
    rel = []
    for i in range(len(path) - 1):
        if path[i + 1] in provider[path[i]]:
            rel.append('c2p')
        elif path[i + 1] in customer[path[i]]:
            rel.append('p2c')
        else:
            rel.append('p2p')
    return '|'.join(rel)

valleyfree, novalleyfree = set(), set()
graph = nx.Graph()
with open(RAW_DIR / 'aspaths.txt') as f:
    for line in f:
        path = line.strip().split('|')[::-1]
        valid = True
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) not in relationship and (path[i + 1], path[i]) not in relationship:
                valid = False
        if not valid:
            continue
        index, vf = -1, 0
        for i in range(len(path) - 1):
            if vf == 0:
                if path[i + 1] in peer[path[i]]:
                    vf = 1
                elif path[i + 1] in customer[path[i]]:
                    vf = 2
            elif vf == 1:
                if path[i + 1] in provider[path[i]] or path[i + 1] in peer[path[i]]:
                    index = i
                    break
                elif path[i + 1] in customer[path[i]]:
                    vf = 2
            elif vf == 2:
                if path[i + 1] in provider[path[i]] or path[i + 1] in peer[path[i]]:
                    index = i
                    break
        if index != -1:
            for i in range(index + 1, len(path)):
                novalleyfree.add(tuple(path[:i + 1]))
            path = path[:index + 1]
        valleyfree.add(tuple(path))
        for i in range(len(path) - 1):
            graph.add_edge(path[i], path[i + 1])

alllinks = defaultdict(int)
traffic_in, traffic_out = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
for path in valleyfree:
    for i in range(len(path) - 1):
        alllinks[(path[i], path[i + 1])] += 1
        alllinks[(path[i + 1], path[i])] += 1

        traffic_in[path[i + 1]][path[i]] += 1
        traffic_out[path[i]][path[i + 1]] += 1

fewlinks = set()
for link in alllinks:
    (asn1, asn2) = link
    asn_in, asn_out = traffic_in[asn2], traffic_out[asn1]
    all_in, all_out = sum(list(asn_in.values())) + 0.001, sum(list(asn_out.values())) + 0.001
    if alllinks[link] < 5 and asn_in[asn1] / all_in < 0.05 and asn_out[asn2] / all_out < 0.05 and (graph.degree(asn1) > 100 or graph.degree(asn2) > 100):
        fewlinks.add(link)

remove_links = set()
for link in fewlinks:
    (asn1, asn2) = link
    if (asn1, asn2) in fewlinks and (asn2, asn1) in fewlinks:
        remove_links.add((asn1, asn2))
        remove_links.add((asn2, asn1))

main_links = set()
for link in graph.edges():
    if (link[0], link[1]) not in remove_links:
        main_links.add((link[0], link[1]))
        main_links.add((link[1], link[0]))

newgraph = copy.deepcopy(graph)
for edge in newgraph.edges():
    (asn1, asn2) = edge
    if (asn1, asn2) not in main_links and (newgraph.degree(asn1) > 100 or newgraph.degree(asn2) > 100):
        graph.remove_edge(asn1, asn2)

print(len(valleyfree))
INTERMEDIATE_DIR.mkdir(exist_ok=True)
fout = open(INTERMEDIATE_DIR / 'done_rib.txt', 'w')
for path in valleyfree:
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) not in graph.edges():
            break
    else:
        fout.write('|'.join(path[::-1]) + '\n')
fout.close()

fout = open(INTERMEDIATE_DIR / 'asrel_rib.txt', 'w')
for edge in graph.edges():
    (asn1, asn2) = edge
    if (asn1, asn2) in relationship:
        fout.write(asn1 + '|' + asn2 + '|' + relationship[(asn1, asn2)] + '\n')
    elif (asn2, asn1) in relationship:
        fout.write(asn2 + '|' + asn1 + '|' + relationship[(asn2, asn1)] + '\n')
fout.close()
