import numpy as np
import matplotlib.pyplot as plt

num_cuts = 40
num_requests = 100

file_path = f'loss_{num_cuts}_{num_requests}.npy'
losses = np.load(file_path, allow_pickle=True)


plt.figure()
for loss in losses:
    plt.plot(loss)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Minimum Loss: {np.min(np.concatenate(losses))}')
plt.legend([f'Min Loss: {np.min(losses[i]):.2f}' for i in range(len(losses))])
plt.savefig(f'loss_{num_cuts}_{num_requests}.png')
plt.show()