import matplotlib, sklearn, pickle, sys
from sklearn.metrics import roc_curve, auc

matplotlib.use("agg") # We use a non-interactive backend as we are generating PNG files, reduces dependencies
import matplotlib.pyplot as plt

print("[ ] Loading the model...")

try:
    model = pickle.loads(open("model.bin", "rb").read())
except FileNotFoundError:
    print("[!] Error: model.bin not found. './run model' to build it.")
    sys.exit(-1)

y_score = model["model"].decision_function(model["x_test"])

false_positives, true_positives, _ = roc_curve(model["y_test"] , y_score)

# ROC curve
plt.figure()

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.plot(false_positives, true_positives, color="red")

print("[ ] Saving plots/roc.png")
plt.savefig("plots/roc.png")

print("[+] Done.")
