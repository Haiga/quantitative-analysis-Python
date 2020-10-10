from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import math
from quantitativo.l2r.utils import load_L2R_file
from quantitativo.l2r.measures10 import modelEvaluation, getGeoRisk, getTRisk, getRisk
from scipy.stats import variation, spearmanr, pearsonr


class dataset:
    def __init__(self):
        self.q = None
        self.x = None
        self.y = None


# coll = "2003_td_dataset"
# num_features = 64
coll = "web10k"
num_features = 136


num_replicacoes = 100
fold = 1
mask = "1" * num_features
test = dataset()
train = dataset()
# testFile = "E:/BCC/Disciplinas Faculdade/TCC/tcc_l2r/dataset/" + coll + "/Fold" + str(fold) + "/Norm.test.txt"
if 'web10k' in coll:
    trainFile = "E:/BCC/Disciplinas Faculdade/TCC/tcc_l2r/dataset/" + coll + "/Fold" + str(fold) + "/train.txt"
else:
    trainFile = "E:/BCC/Disciplinas Faculdade/TCC/tcc_l2r/dataset/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"
# test.x, test.y, test.q = load_L2R_file(testFile, mask)
train.x, train.y, train.q = load_L2R_file(trainFile, mask)

vs = []
end = 6
if 'web10k' in coll:
    f = open('./cluster/web10k/result.txt', 'w+')
else:
    f = open('./cluster/td/result.txt', 'w+')
for i in range(2, end):
    my_train = np.copy(train.x)
    print('.')
    print(i)
    kmeans = KMeans(n_clusters=i, random_state=0).fit(my_train)

    y = kmeans.labels_
    # x = kmeans.predict(my_train[0])
    #     # x = kmeans.predict(my_train[1])

    v = silhouette_score(my_train, y)
    vs.append(v)
    print(v)
    f.write(str(i) + " " + "{:.4f}".format(v) + "\n")
f.close()
import matplotlib.pyplot as plt

plt.plot(range(2, end), vs)
plt.show()

# tentar clusterizar o web10k com as features de 0 a 94 ou 94 ao final, ou 0 a 9

# plotar algum gráfico de nº docs relevantes para essas cluster, bar plot

# se nada disso funcionar fazer cluster separando por num docs relevantes
