import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def factor_pairs(x):
    min_pairs = 5

    def get_factor_pairs(n):
        pairs = []
        limit = int(abs(n) ** 0.5) + 1
        for i in range(1, limit):
            if n % i == 0 and i != 1:
                pairs.append((i, n // i))
        return pairs

    pairs = get_factor_pairs(x)

    offset = 1
    while len(pairs) < min_pairs:
        higher_pairs = get_factor_pairs(x + offset)
        for pair in higher_pairs:
            if (1 not in pair) and (pair not in pairs):
                pairs.append(pair)
        offset += 1

    return pairs


def calculate_piece_areas(x_cuts, y_cuts):
    x_coords = np.sort(np.concatenate(([0], x_cuts, [width])))
    y_coords = np.sort(np.concatenate(([0], y_cuts, [height])))

    piece_widths = np.diff(x_coords)
    piece_heights = np.diff(y_coords)

    areas = np.concatenate(np.outer(piece_widths, piece_heights))

    return areas


def loss_function(areas, requests):
    R = requests
    V = areas

    num_requests = len(R)
    num_values = len(V)

    cost_matrix = np.zeros((num_requests, num_values))

    for i, r in enumerate(R):
        for j, v in enumerate(V):
            cost_matrix[i][j] = abs(r - v) / r

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return total_cost


def calculate_gradient(x_cuts, y_cuts, requests, curr_loss, epsilon=1e-3):
    grad_x_cuts = np.zeros_like(x_cuts, dtype=float)
    grad_y_cuts = np.zeros_like(y_cuts, dtype=float)

    for i in range(len(x_cuts)):
        x_cuts_eps = x_cuts.copy()
        x_cuts_eps[i] += epsilon
        areas_eps = calculate_piece_areas(x_cuts_eps, y_cuts)
        loss_eps = loss_function(areas_eps, requests)
        grad_x_cuts[i] = (loss_eps - curr_loss) / epsilon

    for i in range(len(y_cuts)):
        y_cuts_eps = y_cuts.copy()
        y_cuts_eps[i] += epsilon
        areas_eps = calculate_piece_areas(x_cuts, y_cuts_eps)
        loss_eps = loss_function(areas_eps, requests)
        grad_y_cuts[i] = (loss_eps - curr_loss) / epsilon

    return grad_x_cuts, grad_y_cuts


def gradient_descent(
    factors,
    requests,
    learning_rate=0.1,
    num_iterations=200,
    epsilon=1e-3,
    learning_rate_decay=1,
    num_restarts=10,
):
    best_loss = float("inf")
    best_x_cuts = None
    best_y_cuts = None
    all_losses = []

    for factor in factors:
        # print(f"Factor pair: {factor}")
        num_horizontal, num_vertical = factor

        restart_losses = []

        for i in range(num_restarts):
            x_cuts = np.array(np.random.randint(1, width, num_vertical), dtype=float)
            y_cuts = np.array(np.random.randint(1, height, num_horizontal), dtype=float)

            best_x_cuts = x_cuts.copy()
            best_y_cuts = y_cuts.copy()

            losses = []
            lr = learning_rate
            for i in range(num_iterations):
                lr = max(lr * learning_rate_decay, 1e-2)

                areas = calculate_piece_areas(x_cuts, y_cuts)
                loss = loss_function(areas, requests)
                losses.append(loss)

                if loss < best_loss:
                    best_loss = loss
                    best_x_cuts = x_cuts.copy()
                    best_y_cuts = y_cuts.copy()

                grad_x_cuts, grad_y_cuts = calculate_gradient(
                    x_cuts, y_cuts, requests, loss, epsilon
                )

                x_cuts -= lr * grad_x_cuts
                y_cuts -= lr * grad_y_cuts
                # print(f'Iteration {i + 1}: Loss = {loss}, Best loss = {best_loss}')
            restart_losses.append(losses)
        all_losses.append(restart_losses)
    all_losses = np.array(all_losses)

    return best_x_cuts, best_y_cuts, all_losses


requests = {
    "g10": [
        34,
        23,
        79,
        34,
        51,
        88,
        61,
        80,
        24,
        15,
        14,
        51,
        67,
        75,
        64,
        64,
        60,
        46,
        11,
        95,
        75,
        79,
        23,
        91,
        65,
        82,
        56,
        91,
        89,
        89,
        88,
        87,
        72,
        21,
        26,
        31,
        62,
        26,
        43,
        40,
        69,
        54,
        64,
        67,
        80,
        90,
        24,
        16,
        38,
        88,
        44,
        72,
        78,
        58,
        84,
        83,
        32,
        65,
        51,
        72,
        72,
        66,
        17,
        47,
        97,
        91,
        70,
        67,
        59,
        99,
        36,
        72,
        96,
        10,
        37,
        99,
        18,
        72,
        92,
        46,
        97,
        13,
        85,
        18,
        10,
        19,
        76,
    ],
    "g11": [
        99,
        34,
        96,
        52,
        19,
        19,
        43,
        24,
        19,
        26,
        48,
        70,
        33,
        23,
        80,
        90,
        25,
        11,
        29,
        72,
        87,
        62,
        53,
        41,
        79,
        22,
        84,
        78,
        24,
        61,
        65,
        50,
        15,
        26,
        75,
        15,
        70,
        58,
        65,
        96,
        35,
        28,
        45,
        28,
        34,
        95,
        32,
        51,
        87,
        27,
        32,
        62,
        46,
        24,
        33,
        58,
        29,
        29,
        44,
        95,
        54,
        71,
        35,
        27,
        52,
        54,
        58,
        90,
        73,
        26,
        68,
        26,
        39,
        52,
        52,
        31,
        50,
        89,
        21,
        50,
        14,
        73,
        32,
        42,
        66,
    ],
    "g12": [
        16.25,
        14.16,
        21.58,
        20.36,
        11.69,
        18.41,
        16.13,
        15.3,
        25.46,
        26.64,
        15.54,
        16.64,
        12.22,
        27.98,
        16.28,
        12.92,
        21.9,
        10.23,
        10.0,
        18.8,
        10.91,
        21.85,
        18.72,
        14.06,
        13.89,
        10.98,
        12.42,
        19.53,
        24.25,
        12.61,
        29.61,
        14.27,
        22.86,
        12.89,
        17.4,
        17.6,
        10.0,
        15.82,
        10.0,
        21.78,
        90.0,
        89.68,
        87.57,
        83.64,
        90.0,
        74.41,
        86.07,
        84.76,
        73.45,
        84.11,
        84.79,
        87.2,
        89.63,
        86.98,
        87.1,
        90.0,
        89.48,
        85.83,
        86.22,
        77.94,
        75.5,
        81.31,
        80.44,
        90.0,
        88.94,
        85.1,
        78.96,
        80.9,
        75.74,
        85.71,
        90.0,
        83.16,
        84.42,
        87.58,
        75.54,
        74.13,
        85.62,
        74.22,
        77.43,
        89.32,
    ],
    "g13": [
        92.78,
        86.94,
        97.09,
        94.4,
        90.67,
        37.64,
        92.89,
        95.06,
        89.2,
        92.01,
        91.73,
        88.93,
        83.96,
        87.75,
        12.89,
        95.03,
        89.19,
        93.45,
        73.84,
        94.24,
        93.45,
        89.04,
        92.3,
        87.02,
        85.73,
        28.93,
        90.48,
        10.52,
        91.22,
        85.61,
        94.46,
        10.18,
        93.42,
        91.13,
        89.79,
        32.19,
        89.8,
        94.97,
        84.77,
        89.83,
        94.4,
        85.73,
        95.7,
        40.66,
        86.39,
        69.65,
        92.0,
        12.36,
        90.25,
        94.09,
        91.48,
        26.81,
        99.61,
        10.7,
        94.41,
        90.81,
        95.78,
        91.65,
        92.54,
        95.53,
        37.49,
        86.59,
        91.06,
        91.09,
        89.06,
        12.54,
        56.17,
        82.88,
        86.1,
        98.78,
        88.33,
        95.4,
        93.64,
        94.54,
        94.1,
        10.13,
        95.84,
        92.07,
        91.59,
        95.22,
        86.3,
        13.87,
        87.21,
        87.19,
        89.69,
        10.09,
        93.3,
        89.78,
        81.45,
        54.29,
        90.73,
        82.97,
        91.01,
        12.86,
        84.61,
        87.76,
        82.61,
    ],
    "g14": [
        90.3,
        92.67,
        87.18,
        87.25,
        86.88,
        85.44,
        89.68,
        88.58,
        84.93,
        90.83,
        91.76,
        92.6,
        90.44,
        81.14,
        88.88,
        86.61,
        90.61,
        95.55,
        80.0,
        86.67,
        82.78,
        97.1,
        84.65,
        90.99,
        98.51,
        87.36,
        92.48,
        83.53,
        84.5,
        87.13,
        92.39,
        90.06,
        89.82,
        80.0,
        88.69,
        91.8,
        85.26,
        88.63,
        80.36,
        92.62,
        100.0,
        80.16,
        85.29,
        89.15,
        92.04,
        92.63,
        98.03,
        86.88,
        83.91,
        95.26,
        90.48,
        89.19,
        89.86,
        86.06,
        91.73,
        97.21,
        84.58,
        91.87,
        87.77,
        98.29,
        94.18,
        84.57,
        84.61,
        85.4,
        90.45,
        81.25,
        85.55,
        87.34,
        84.58,
        91.96,
        89.16,
        88.52,
        92.66,
        92.05,
        96.1,
        87.39,
        47.86,
        44.95,
        55.75,
        40.9,
        54.84,
        52.58,
        47.52,
        45.8,
        58.85,
        56.54,
        45.95,
        44.97,
        55.64,
        46.58,
        47.45,
        50.48,
        56.24,
        58.86,
        49.85,
        46.45,
        59.36,
        42.36,
        44.29,
        47.12,
    ],
    "g15": [
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    ],
    "g16": [
        93.27,
        46.39,
        53.43,
        66.26,
        48.92,
        66.08,
        41.31,
        76.15,
        70.98,
        45.03,
        62.44,
        30.81,
        38.85,
        61.0,
        74.38,
        13.69,
        13.73,
        91.64,
        90.68,
        72.68,
        88.69,
        36.83,
        22.07,
        90.96,
        37.37,
        23.65,
        89.14,
        78.85,
        19.19,
        93.79,
        63.51,
        55.97,
        94.45,
        21.18,
        72.52,
        31.17,
        40.4,
        43.71,
        81.1,
        95.94,
        18.9,
        71.62,
        63.48,
        30.35,
        12.29,
        35.18,
        51.72,
        90.6,
        25.64,
        72.59,
        99.03,
        43.1,
        12.83,
        25.17,
        68.47,
        56.74,
        91.39,
        63.26,
        67.87,
        87.34,
        77.52,
        43.03,
        39.44,
        10.77,
        67.46,
        29.02,
        43.37,
        63.62,
        98.24,
        67.76,
        40.11,
        24.1,
        71.18,
        33.93,
        61.39,
        52.4,
        29.9,
        69.44,
        86.33,
        63.58,
        49.19,
        93.13,
        40.0,
        37.91,
        97.03,
        96.14,
        28.73,
        13.59,
        86.48,
        81.06,
        62.14,
        66.47,
        83.87,
        92.07,
        45.07,
        68.2,
        11.44,
        61.3,
        69.74,
        84.08,
    ],
}

for request_name, request in requests.items():
    request = np.array(request, dtype=float)
    
    height = round(math.sqrt(1.05 * np.sum(request) / 1.6), 2)
    width = round(height * 1.6, 2)

    learning_rate = 1
    learning_rate_decay = 0.99
    num_iterations = 500
    epsilon = 1e-3
    num_restarts = 10

    factors = factor_pairs(len(request))
    best_x_cuts, best_y_cuts, all_losses = gradient_descent(
        factors, request, learning_rate, num_iterations, epsilon, learning_rate_decay, num_restarts
    )

    fig, axs = plt.subplots(len(factors), 1, figsize=(8, len(factors) * 5))
    for i, (num_horizontal, num_vertical) in enumerate(factors):
        ax = axs[i] if len(factors) > 1 else axs

        for j, losses in enumerate(all_losses[i]):
            ax.plot(losses, label=f"Restart {j}")

        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title(f"{request_name}: {num_horizontal}x{num_horizontal}, best loss: {np.min(np.concatenate(all_losses[i]))}")
        ax.legend()

    print("Best x cuts:", best_x_cuts)
    print("Best y cuts:", best_y_cuts)

    fig.tight_layout()
    plt.savefig(f"{request_name}_{np.min(np.concatenate(all_losses))}.png")
    np.save(f"{request_name}.npy", [all_losses, best_x_cuts, best_y_cuts])
