import numpy as np
import matplotlib.pyplot as plt

PCA = (0.98, 0.67, 0.79)
LSTM = (0.9526, 0.9903, 0.9711)

fig, ax = plt.subplots()

index = np.arange(3)
bar_width = 0.2
opacity = 0.4

rects1 = ax.bar(index, PCA, bar_width, alpha=opacity, color='b', label='PCA')
rects2 = ax.bar(index + bar_width, LSTM, bar_width, alpha=opacity, color='r', label='LSTM')

ax.set_xlabel('Measure')
ax.set_ylabel('Scores')
ax.set_title('Scores by different models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Precesion', 'Recall', 'F1-score'))
ax.legend()
plt.show()

x = [8, 9, 10, 11]
FP = [605, 588, 495, 860]
FN = [465, 333, 108, 237]
TP = [4123 - FN[i] for i in range(4)]
P = [TP[i] / (TP[i] + FP[i]) for i in range(4)]
R = [TP[i] / (TP[i] + FN[i]) for i in range(4)]
F1 = [2 * P[i] * R[i] / (P[i] + R[i]) for i in range(4)]
l1 = plt.plot(x, P, ':rx')
l2 = plt.plot(x, R, ':b+')
l3 = plt.plot(x, F1, ':k^')
plt.xlabel('window_size')
plt.ylabel('scores')
plt.legend((l1[0], l2[0], l3[0]), ('Precision', 'Recall', 'F1-score'))
plt.xticks(x)
plt.ylim((0.5, 1))
plt.show()
