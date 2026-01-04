import matplotlib.pyplot as plt

classes = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
f1_scores = [0.4915, 0.8034, 0.6620, 0.6316, 0.6954, 0.8905, 0.5276, 0.7895]

plt.figure(figsize=(6, 4))
plt.bar(classes, f1_scores)
plt.ylim(0, 1.0)
plt.xlabel("Class")
plt.ylabel("F1-score")
plt.title("F1-score per class on the test set")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("fig_f1_per_class.png", dpi=300)
plt.close()