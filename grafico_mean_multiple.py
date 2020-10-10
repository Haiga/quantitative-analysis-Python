import numpy as np
import math
import matplotlib.pyplot as plt

f = open('temp4/result.spearman.txt')
num_replicacoes = 100
values = []
cont = 0
for line in f:
    if cont != 0 and cont <= num_replicacoes:
        # if cont != 0:
        #     values.append([math.log(float(x)) if float(x) > 0 else -math.log(-float(x)) for x in line.replace('\n', '').split('\t')])
        values.append([float(x) for x in line.replace('\n', '').split('\t')])
    cont += 1
f.close()

print(values)
values = np.array(values)
print(np.mean(values.T, axis=1))
medias = np.mean(values.T, axis=1)
media_medias = np.mean(medias)
efeito = medias - media_medias

num_sistemas = 9

# z = np.argsort(values[:, 2])
x = [j for j in range(num_replicacoes)]
# for i in range(num_sistemas):
#     plt.plot(x, values[:, i])

fig = plt.figure()
ax = fig.add_subplot(111)
labels = ['Georisk-NDCG', 'Georisk-MAP', 'Georisk-MRR', 'TRisk-NDCG', 'TRisk-MAP', 'TRisk-MRR', 'Risk-NDCG', 'Risk-MAP',
          'Risk-MRR']

ax.plot(x, values[:, 0], c='k', marker=".", ls='-', label=labels[0])
ax.plot(x, values[:, 1], c='k', marker="+", ls='-', label=labels[1])
ax.plot(x, values[:, 2], c='#006666', marker="v", ls='-', label=labels[2])

ax.plot(x, values[:, 3], c='k', marker=".", ls=':', label=labels[3])
ax.plot(x, values[:, 4], c='k', marker="+", ls=':', label=labels[4])
ax.plot(x, values[:, 5], c='k', marker="v", ls=':', label=labels[5])

ax.plot(x, values[:, 6], c='k', marker=".", ls='--', label=labels[6])
ax.plot(x, values[:, 7], c='#008800', marker="+", ls='--', label=labels[7])
ax.plot(x, values[:, 8], c='k', marker="v", ls='--', label=labels[8])


plt.xticks([0, 4, 9, 14,  19], ['1', '5', '10', '15',  '20'])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=3, fancybox=True, shadow=True)
plt.xlabel('Replicações')
plt.ylabel('Correlação c/ Variância')
plt.show()
