from om.ont import get_n, tokenize

from rdflib import Graph
from rdflib.term import URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL
import re
from utils import gn
import dill
import torch
import random


def build_dataset1(base, schema, result_path):
    yago = Graph().parse(base) + Graph().parse(schema)
    yago.bind('yago-knowledge', 'http://yago-knowledge.org/resource/')
    yago.bind('schema', 'http://schema.org/')
    yago.bind('bioschema', 'http://bioschemas.org/')
    yago.bind('custom', 'http://custom.org/')
    for s, p, o in yago:
        if type(o) is Literal and o.language != 'en':
            yago.remove((s, p, o))

    with open(result_path, 'rb') as f:
        result = dill.loads(f.read())
        print(len(result))

    l = []
    x = []
    y = []

    cs = set()

    for r, c in result:

        if (r, RDFS.comment, None) not in yago:
            continue

        cm = yago.value(r, RDFS.comment)

        if cm in cs or len(re.sub(r'[^A-Za-z]+', '', cm)) <= 0:
            continue

        cs.add(cm)

        l.append(' '.join(tokenize(gn(r, yago))).lower())
        x.append(str(cm))
        y.append(c)

    y = torch.LongTensor(y)
    return l, x, y


def build_dataset2(base):
    with open(base, 'r') as f:
        raw_data = list(map(lambda x: x[:-1], f.readlines()))

    tm = {'abstract': 0, 'endurant': 1, 'perdurant': 2, 'quality': 3, 'situation': 4}
    ts = {x: [] for x in range(5)}

    for s in raw_data:
        line = s.split(';')
        x = line[-1]
        c = line[1]
        y = tm[line[0]]

        ts[y].append((x, c, y))

    mv = min(map(len, list(ts.values())))
    print(mv)

    nd = []

    for s in ts.values():
        random.shuffle(s)
        nd.extend(random.choices(s, k=mv))

    c, x, y = list(zip(*nd))

    y = torch.LongTensor(y)

    return c, x, y
