import random

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from quantitativo.quantitativeAnalysis.l2r.utils import load_L2R_file
from quantitativo.quantitativeAnalysis.l2r.measures10 import relevanceTest


class dataset:
    def __init__(self):
        self.q = None
        self.x = None
        self.y = None


# coll = "2003_td_dataset"
# num_features = 64
coll = "web10k"
num_features = 136

fold = 1

mask = "1" * num_features
test = dataset()
train = dataset()

if 'web10k' in coll:
    trainFile = "E:/BCC/Disciplinas Faculdade/TCC/tcc_l2r/dataset/" + coll + "/Fold" + str(fold) + "/train.txt"
    # trainFile = "./dataset/" + coll + "/Fold" + str(fold) + "/train.txt"
else:
    trainFile = "E:/BCC/Disciplinas Faculdade/TCC/tcc_l2r/dataset/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"
    # trainFile = "./dataset/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"

train.x, train.y, train.q = load_L2R_file(trainFile, mask)


num_amostras = train.x.shape[0]
num_algoritms = train.x.shape[1]

# my_slice_docs = random.sample(range(num_amostras), int(0.5 * num_amostras))
# my_slice_docs = np.sort(my_slice_docs)
#
# temp_dataset = dataset()
# temp_dataset.x = train.x[my_slice_docs, :]
# temp_dataset.y = train.y[my_slice_docs]
# temp_dataset.q = train.q[my_slice_docs]

#Contar o número de documentos relevantes por query

f = open('./cluster/web_hist_docs_relevantes.txt', 'w+')

anterior = -1
total_relevantes = 0
start_anterior = 0
allConts = []
start = True
for i in range(num_amostras):
    if train.q[i] != anterior:
        if not start:
            allConts.append(total_relevantes)
            f.write(str(total_relevantes) + "\n")
        start = False
        total_relevantes = 0
        anterior = train.q[i]
        start_anterior = i
    else:
        r = relevanceTest(coll, train.y[i])
        if r == 1:
            total_relevantes += 1

f.close()
