import pickle
import numpy as np


def load_radioml(path):

    with open(path, 'rb') as f:
        # file was saved with Python 2, so latin1 is needed to read it
        data = pickle.load(f, encoding="latin1")

    #Will the store the arrays containing radio signals
    signal_groups = []
    #Will store the corresponding modulation name
    mods = []
    #Will store the SNR of each signal
    snrs = []

    for (modulation, snr), signals in data.items():
        signal_groups.append(signals)
        mods.extend([modulation] * len(signals))
        snrs.extend([snr] * len(signals))

    X = np.concatenate(signal_groups, axis=0)

    return X, np.array(mods), np.array(snrs)