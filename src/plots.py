import numpy as np
import matplotlib.pyplot as plt


def plot_signal(X, mods, snrs, mod, snr):
    # Returns the indices that match the conditions
    indices = np.where((mods == mod) & (snrs == snr))[0]

    # Select the first radio signal
    signal = X[indices[0]]

    # Extract I and Q
    I = signal[0]
    Q = signal[1]

    # Create sample indices
    samples = np.arange(128)

    # Plot
    plt.figure(figsize=(12, 4))

    plt.plot(samples, I, label="I")
    plt.plot(samples, Q, label="Q")

    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

    plt.title(
        f"Radio Signal: {mod}, SNR: {snr} dB"
    )

    plt.legend()
    plt.grid(True)

    plt.show()




def plot_constellation(X, mods, snrs, mod, snr):
    # Returns the indices that match the conditions
    indices = np.where((mods == mod) & (snrs == snr))[0]

    # Select the first radio signal
    signal = X[indices[0]]

    # Extract I and Q
    I = signal[0]
    Q = signal[1]

    # Plot
    plt.figure(figsize=(6, 6))

    plt.scatter(I, Q, s=20)

    plt.xlabel("I")
    plt.ylabel("Q")

    plt.title(f"{mod} Constellation at {snr} dB")

    plt.grid(True)
    plt.axis("equal")

    plt.show()


