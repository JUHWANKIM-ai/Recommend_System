import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Agg 백엔드 설정 (GUI 없이 실행 가능)


def plot_hist(epochs, hist_numpy, title):
    plt.figure(figsize=(15, 3), dpi=400)
    plt.plot(range(1, epochs + 1), hist_numpy.iloc[:, 0], label="train")
    plt.plot(range(1, epochs + 1), hist_numpy.iloc[:, 1], label="val")
    plt.title(f"Train-Val {title}")
    plt.xlabel("Training Epochs")
    plt.ylabel(f"{title}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"results/figures/{title}_plot.png", bbox_inches='tight')
