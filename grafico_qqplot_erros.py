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
        values.append(
            [math.log(float(x)) if float(x) > 0 else -math.log(-float(x)) for x in line.replace('\n', '').split('\t')])
        # values.append([float(x) for x in line.replace('\n', '').split('\t')])
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


#
erros = []
for i in range(num_sistemas):
    for j in range(num_replicacoes):
        # plt.plot(medias[i], values[j][i] - media_medias - efeito[i], '.', markersize=6, color='#004545')
        erros.append(values[j][i] - media_medias - efeito[i])


X_100 = []
for i in range(1, 101):
    X_100.append(np.percentile(erros, i))

X_100 = np.array(X_100)

Y = np.random.normal(loc=0, scale=1, size=1000)
Y_100 = []
for i in range(1, 101):
    Y_100.append(np.percentile(Y, i))

# plt.scatter(X_100, Y_100)
plt.scatter(Y_100, X_100)
plt.ylabel("Residual Quantile")
plt.xlabel("Normal Quantile")

m, b = np.polyfit(X_100, Y_100, 1)

# plt.plot(X_100, m * X_100 + b)
plt.plot(m * X_100 + b, X_100)

plt.show()
