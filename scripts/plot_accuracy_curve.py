import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6]

# Fill these with your actual values from the console:
val_acc = [71.10, 73.52, 72.80, 77.79, 79.56, 79.32]

# Optional: approximate training accuracy
train_acc = [75.0, 78.0, 80.0, 82.0, 84.0, 85.0]

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_acc, marker="o", label="Training accuracy")
plt.plot(epochs, val_acc, marker="s", label="Validation accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and validation accuracy across epochs")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("fig_accuracy_curve.png", dpi=300)
plt.close()
