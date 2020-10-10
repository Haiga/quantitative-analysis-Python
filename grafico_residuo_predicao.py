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

# for i in range(5):
#     plt.plot(medias, (values.T - media_medias)[:, i] - efeito, 'o')



for i in range(num_sistemas):
    for j in range(num_replicacoes):
        plt.plot(medias[i], values[j][i] - media_medias - efeito[i], '.', markersize=6, color='#004545')

# plt.xlim((-1, -0.8))
# plt.ylim((-0.025, 0.045))
# # plt.ylim((-0.9, 0.9))
# plt.ylim((-0.5, 0.5))

# plt.hlines(0, -10, 10, linewidth=1, linestyles='--')
# plt.hlines(0, -10, 10, linewidth=1)
plt.xlabel('Predição de Resposta')
plt.ylabel('Resíduo / erro')

plt.show()





