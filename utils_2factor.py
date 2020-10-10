import sklearn
import math
from quantitativo.l2r.utils import load_L2R_file
from quantitativo.l2r.measures import modelEvaluation, getGeoRisk, getTRisk, getRisk
from scipy.stats import variation, spearmanr, pearsonr


class dataset:
    def __init__(self):
        self.q = None
        self.x = None
        self.y = None


def std_dev_amostra(vec):
    vec = np.array(vec)
    media = np.mean(vec)
    diff = (vec - media)
    s = np.sum(diff * diff)
    s = s / (len(vec) - 1)
    return math.pow(s, 0.5)


coll = "2003_td_dataset"
num_replicacoes = 5

fold = 1
num_features = 64
mask = "1" * num_features
test = dataset()
train = dataset()
# testFile = "E:/BCC/Disciplinas Faculdade/TCC/tcc_l2r/dataset/" + coll + "/Fold" + str(fold) + "/Norm.test.txt"
trainFile = "E:/BCC/Disciplinas Faculdade/TCC/tcc_l2r/dataset/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"
# test.x, test.y, test.q = load_L2R_file(testFile, mask)
train.x, train.y, train.q = load_L2R_file(trainFile, mask)

import numpy as np
import random

num_amostras = train.x.shape[0]
num_algoritms = train.x.shape[1]

for r in range(num_replicacoes):
    fp = open('./temp/' + str(r) + '.pearson' + '.txt', 'w+')
    fs = open('./temp/' + str(r) + '.spearman' + '.txt', 'w+')
    fp.write(' \tNDCG\tMAP\tMRR\n')
    fs.write(' \tNDCG\tMAP\tMRR\n')

    pfp = open('./temp/' + str(r) + '.p.pearson' + '.txt', 'w+')
    pfs = open('./temp/' + str(r) + '.p.spearman' + '.txt', 'w+')
    pfp.write(' \tNDCG\tMAP\tMRR\n')
    pfs.write(' \tNDCG\tMAP\tMRR\n')

    my_slice_docs = random.sample(range(num_amostras), int(0.5 * num_amostras))
    my_slice_docs = np.sort(my_slice_docs)

    temp_dataset = dataset()
    temp_dataset.x = train.x[my_slice_docs, :]
    temp_dataset.y = train.y[my_slice_docs]
    temp_dataset.q = train.q[my_slice_docs]

    all_ndcg_queries = []
    all_ap_queries = []
    all_mrr_queries = []
    for i in range(num_algoritms):
        # x , y = modelEvaluation(temp_dataset, train.x[my_slice_docs, i], num_features)
        ndcg_queries, ap_queries, mrr_queries = modelEvaluation(temp_dataset, temp_dataset.x[:, i], num_features)
        all_ndcg_queries.append(ndcg_queries)
        all_ap_queries.append(ap_queries)
        all_mrr_queries.append(mrr_queries)

    # Computes Georisk
    mat_ndcg = np.array(all_ndcg_queries).T
    mat_ap = np.array(all_ap_queries).T
    mat_mrr = np.array(all_mrr_queries).T
    c1 = getGeoRisk(mat_ndcg, 3)
    c2 = getGeoRisk(mat_ap, 3)
    c3 = getGeoRisk(mat_mrr, 3)

    fs.write('Georisk\t')
    fp.write('Georisk\t')
    pfs.write('Georisk\t')
    pfp.write('Georisk\t')

    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    fs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[0]
    fs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    fs.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    fp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[0]
    fp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    fp.write("{:.4f}".format(result) + "\n")

    ######
    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    pfs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[1]
    pfs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    pfs.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    pfp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[1]
    pfp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    pfp.write("{:.4f}".format(result) + "\n")
    ######
    # for i in range(num_algoritms):
    #     c = c1[i]
    #     c = c2[i]
    #     c = c3[i]

    base_ndcg = np.mean(mat_ndcg, axis=1)
    base_ap = np.mean(mat_ap, axis=1)
    base_mrr = np.mean(mat_mrr, axis=1)
    c1 = []
    c2 = []
    c3 = []
    for i in range(num_algoritms):
        c1.append(getTRisk(all_ndcg_queries[i], base_ndcg, 3))
        c2.append(getTRisk(all_ap_queries[i], base_ap, 3))
        c3.append(getTRisk(all_mrr_queries[i], base_mrr, 3))

    fs.write('TRisk\t')
    fp.write('TRisk\t')
    pfs.write('TRisk\t')
    pfp.write('TRisk\t')

    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    fs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[0]
    fs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    fs.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    fp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[0]
    fp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    fp.write("{:.4f}".format(result) + "\n")
    ########
    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    pfs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[1]
    pfs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    pfs.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    pfp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[1]
    pfp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    pfp.write("{:.4f}".format(result) + "\n")
    ########
    c1 = []
    c2 = []
    c3 = []

    for i in range(num_algoritms):
        c1.append(np.mean(getRisk(all_ndcg_queries[i], base_ndcg)))
        c2.append(np.mean(getRisk(all_ap_queries[i], base_ap)))
        c3.append(np.mean(getRisk(all_mrr_queries[i], base_mrr)))

    fs.write('Risk\t')
    fp.write('Risk\t')
    pfs.write('Risk\t')
    pfp.write('Risk\t')

    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[0] * -1
    fs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[0] * -1
    fs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[0] * -1
    fs.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[0] * -1
    fp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[0] * -1
    fp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[0] * -1
    fp.write("{:.4f}".format(result) + "\n")

    ###

    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    pfs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[1]
    pfs.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    pfs.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    pfp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[1]
    pfp.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    pfp.write("{:.4f}".format(result) + "\n")

    fp.close()
    pfp.close()
    fs.close()
    pfs.close()
