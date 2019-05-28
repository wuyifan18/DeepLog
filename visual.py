import numpy as np
import matplotlib.pyplot as plt

PCA = (0.98, 0.67, 0.79)
LSTM = (0.9526, 0.9903, 0.9711)

fig, ax = plt.subplots()

index = np.arange(3)
bar_width = 0.2
opacity = 0.4

rects1 = ax.bar(index, PCA, bar_width,
                alpha=opacity, color='b', label='PCA')

rects2 = ax.bar(index + bar_width, LSTM, bar_width,
                alpha=opacity, color='r', label='LSTM')

ax.set_xlabel('Measure')
ax.set_ylabel('Scores')
ax.set_title('Scores by different models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Precesion', 'Recall', 'F1-score'))
ax.legend()
plt.show()

x = [8, 9, 10, 11]
precision = [0.84313, 0.85652, 0.86877, 0.88274]
recall = [0.92554, 0.90492, 0.87509, 0.84356]
f1 = [0.88241, 0.88006, 0.87192, 0.86271]
l1 = plt.plot(x, precision, ':rx')
l2 = plt.plot(x, recall, ':b+')
l3 = plt.plot(x, f1, ':k^')
plt.xlabel('Top candidate')
plt.ylabel('scores')
plt.legend((l1[0], l2[0], l3[0]), ('Precision', 'Recall', 'F1-score'))
plt.xticks([8, 9, 10, 11])
plt.ylim((0.5, 1))
plt.show()
