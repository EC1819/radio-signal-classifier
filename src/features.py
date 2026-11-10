import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def amplitude_features(x):

    I = x[0]
    Q = x[1]

    amplitude = np.sqrt(I**2 + Q**2)

    amplitude_mean = np.mean(amplitude)
    amplitude_std = np.std(amplitude)
    amplitude_skew = skew(amplitude)
    amplitude_kurtosis = kurtosis(amplitude)
    amplitude_peak_ratio = amplitude.max()/ (amplitude_mean)


    return {
        "amplitude_std": amplitude_std,
        "amplitude_skew": amplitude_skew,
        "amplitude_kurtosis": amplitude_kurtosis,
        "amplitude_peak_ratio": amplitude_peak_ratio
    }


def phase_features(x):

    I = x[0]
    Q = x[1]

    phase = np.arctan2(Q, I)

    return {
        "phase_mean": np.mean(phase),
        "phase_std": np.std(phase),
        "phase_skew": skew(phase),
        "phase_kurtosis": kurtosis(phase)
    }


def frequency_features(x):

    I = x[0]
    Q = x[1]

    phase = np.arctan2(Q, I)
    frequency = np.diff(np.unwrap(phase))

    return {
        "frequency_mean": np.mean(frequency),
        "frequency_std": np.std(frequency),
        "frequency_skew": skew(frequency),
        "frequency_kurtosis": kurtosis(frequency)
    }


def extract_all_features(X):

    I = X[:, 0, :]
    Q = X[:, 1, :]

    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    frequency = np.diff(np.unwrap(phase, axis=1), axis=1)

    amplitude_mean = np.mean(amplitude, axis=1)

    return pd.DataFrame({
        "amplitude_std": np.std(amplitude, axis=1),
        "amplitude_skew": skew(amplitude, axis=1),
        "amplitude_kurtosis": kurtosis(amplitude, axis=1),
        "amplitude_peak_ratio": np.max(amplitude, axis=1) / amplitude_mean,

        "phase_mean": np.mean(phase, axis=1),
        "phase_std": np.std(phase, axis=1),
        "phase_skew": skew(phase, axis=1),
        "phase_kurtosis": kurtosis(phase, axis=1),

        "frequency_mean": np.mean(frequency, axis=1),
        "frequency_std": np.std(frequency, axis=1),
        "frequency_skew": skew(frequency, axis=1),
        "frequency_kurtosis": kurtosis(frequency, axis=1)
    })
