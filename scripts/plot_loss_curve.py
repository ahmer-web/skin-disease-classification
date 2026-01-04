import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6]

val_loss = [0.8143, 0.7293, 0.7581, 0.6227, 0.5932, 0.6209]

# Optional approximate training loss:
train_loss = [1.60, 0.88, 0.76, 0.62, 0.60, 0.56]

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, marker="o", label="Training loss")
plt.plot(epochs, val_loss, marker="s", label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss across epochs")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("fig_loss_curve.png", dpi=300)
plt.close()
