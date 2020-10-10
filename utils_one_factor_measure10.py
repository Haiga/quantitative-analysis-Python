import sklearn
import math
from quantitativo.l2r.utils import load_L2R_file
from quantitativo.l2r.measures10 import modelEvaluation, getGeoRisk, getTRisk, getRisk
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
num_replicacoes = 100

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

p1f = open('./temp4/' + 'result.pearson' + '.txt', 'w+')
s1f = open('./temp4/' + 'result.spearman' + '.txt', 'w+')
p1f.write('GeoriskNDCG\tGeoriskMAP\tGeoriskMRR\tTRiskNDCG\tTRiskMAP\tTRiskMRR\tRiskNDCG\tRiskMAP\tRiskMRR\n')
s1f.write('GeoriskNDCG\tGeoriskMAP\tGeoriskMRR\tTRiskNDCG\tTRiskMAP\tTRiskMRR\tRiskNDCG\tRiskMAP\tRiskMRR\n')
p1f_p = open('./temp4/pvalue/' + 'result.pearson' + '.txt', 'w+')
s1f_p = open('./temp4/pvalue/' + 'result.spearman' + '.txt', 'w+')
p1f_p.write('GeoriskNDCG\tGeoriskMAP\tGeoriskMRR\tTRiskNDCG\tTRiskMAP\tTRiskMRR\tRiskNDCG\tRiskMAP\tRiskMRR\n')
s1f_p.write('GeoriskNDCG\tGeoriskMAP\tGeoriskMRR\tTRiskNDCG\tTRiskMAP\tTRiskMRR\tRiskNDCG\tRiskMAP\tRiskMRR\n')

for r in range(num_replicacoes):
    print(r)

    my_slice_docs = random.sample(range(num_amostras), int(0.4 * num_amostras))
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
        # m_ndcg = max(ndcg_queries)
        # m_ap = max(ap_queries)
        # m_mrr = max(mrr_queries)
        # if m_ndcg > 0 :
        #     ndcg_queries = (ndcg_queries - np.min(ndcg_queries))/m_ndcg
        # if m_ap > 0 :
        #     ap_queries = (ap_queries - np.min(ap_queries))/m_ap
        # if m_mrr > 0 :
        #     mrr_queries = (mrr_queries - np.min(mrr_queries))/m_mrr

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



    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    s1f.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[0]
    s1f.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    s1f.write("{:.4f}".format(result) + "\t")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    p1f.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[0]
    p1f.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    p1f.write("{:.4f}".format(result) + "\t")

    ######
    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
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



    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    s1f.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[0]
    s1f.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    s1f.write("{:.4f}".format(result) + "\t")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[0]
    p1f.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[0]
    p1f.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[0]
    p1f.write("{:.4f}".format(result) + "\t")
    ########
    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
    ########
    c1 = []
    c2 = []
    c3 = []

    for i in range(num_algoritms):
        c1.append(np.mean(getRisk(all_ndcg_queries[i], base_ndcg)))
        c2.append(np.mean(getRisk(all_ap_queries[i], base_ap)))
        c3.append(np.mean(getRisk(all_mrr_queries[i], base_mrr)))


    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[0] * -1
    s1f.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[0] * -1
    s1f.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[0] * -1
    s1f.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[0] * -1
    p1f.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[0] * -1
    p1f.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[0] * -1
    p1f.write("{:.4f}".format(result) + "\n")

    ###

    result = spearmanr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c2, np.nan_to_num(variation(mat_ap)))[1]
    s1f_p.write("{:.4f}".format(result) + "\t")
    result = spearmanr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    s1f_p.write("{:.4f}".format(result) + "\n")

    result = pearsonr(c1, np.nan_to_num(variation(mat_ndcg)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c2, np.nan_to_num(variation(mat_ap)))[1]
    p1f_p.write("{:.4f}".format(result) + "\t")
    result = pearsonr(c3, np.nan_to_num(variation(mat_mrr)))[1]
    p1f_p.write("{:.4f}".format(result) + "\n")

s1f_p.close()
s1f.close()
p1f.close()
p1f_p.close()
