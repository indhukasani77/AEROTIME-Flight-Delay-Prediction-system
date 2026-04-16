import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("static/plots", exist_ok=True)

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.savefig("static/plots/precision_recall.png")
plt.close()

plt.plot(x, np.cos(x))
plt.savefig("static/plots/roc_curves.png")
plt.close()

plt.bar(["A","B","C"], [10,20,15])
plt.savefig("static/plots/feature_importance.png")
plt.close()

plt.imshow([[1,2],[3,4]])
plt.savefig("static/plots/confusion_matrix.png")
plt.close()

plt.bar(["Acc","F1"], [0.75,0.68])
plt.savefig("static/plots/rf_metrics.png")
plt.close()

print("DONE")