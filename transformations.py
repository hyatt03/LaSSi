from math import ceil, log2, pi
import numpy as np

def save_raw_fft(E_vec, X, Y, Z, YZ):
    pass

def run_fft(realspace_motion_parts, dt):
    FFT_res = 8
    power_range = 5 # meV
    E_res = 0.01

    # Convolution (form either instrument resolution or just some minimal smoothing)
    conv_sigma = 0.006
    smooth_range_E = 0.0001 # energy range to be smoothed around.

    # Constants
    k_B = 1.38065e-23
    hbar = 1.05459e-34
    meV = 1.602e-22
    g = 2.002
    mu_B = 9.274e-24
    gamma = - g * mu_B / hbar

    X_acc = 0
    Y_acc = 0
    Z_acc = 0
    YZ_acc = 0
    E_vector = 0
    partTot = len(realspace_motion_parts)
    NFFT = 0
    f = 0

    for partNB in range(partTot):
        m_a = realspace_motion_parts[partNB]

        NFFT = 2^ceil(log2(abs(len(m_a)))) * FFT_res
        f = 2 * pi / dt # Sampling frequency
        freq = (f / NFFT) * np.array(list((range(1, int(NFFT / 2))))) # Frequency vector
        dE = f * hbar / (NFFT * meV)
        E_vector = freq * hbar / meV

        freqs = np.fft.rfftfreq(m_a.shape[0], dt)

        X = np.abs(np.fft.rfft(m_a[:, 0]))
        Y = np.abs(np.fft.rfft(m_a[:, 1]))
        Z = np.abs(np.fft.rfft(m_a[:, 2]))

        if not X_acc:
            X_acc = X
            Y_acc = Y
            Z_acc = Z
            YZ_acc = Y + Z
        else:
            X_acc += X
            Y_acc += Y
            Z_acc += Z
            YZ_acc += Y + Z

    save_raw_fft(E_vector, X_acc, Y_acc, Z_acc, YZ_acc)

    # Time to smooth the data
    # @TODO: Implement smoothing

    E_res_index = round(E_res / (f * hbar / (meV * NFFT)))
    L = len(YZ_acc)

    return X, Y, Z, freqs
