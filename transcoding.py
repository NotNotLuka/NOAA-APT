from scipy.signal.windows import kaiser
from scipy.signal import resample, resample_poly, convolve, correlate
from scipy.io import wavfile
import numpy as np
import cv2
import matplotlib.pyplot as plt

f_carrier = 2400
f_pixel = int(2080 / 0.5)
f_sample = 20800
samples_per_pixel = f_sample // f_pixel


def get_sync_frames(syncing=False):
    sync_frame_str = "..WW..WW..WW..WW..WW..WW..WW........"
    sync_frame_alt = np.array([1 if x == "W" else -1 for x in sync_frame_str])

    sync_frame_str0 = (
        "000010001000100010001000100010000000000"
        if syncing
        else "000011001100110011001100110011000000000"
    )
    sync_frame_str1 = "000011100111001110011100111001110011100"
    sync_frame0 = [int(x) * 2 - 1 for x in sync_frame_str0]
    sync_frame1 = [int(x) * 2 - 1 for x in sync_frame_str1]

    return sync_frame0, sync_frame1, sync_frame_alt


# def resample_decimate(data, sr, rec_sr):
#     gcd = np.gcd(sr, rec_sr)
#     L = sr // gcd
#     M = rec_sr // gcd
#     f_cutoff = 0.5 * np.min([1 / M, 1 / L])

#     def sinc(n):
#         return 2 * f_cutoff * np.sinc(2 * f_cutoff * n)

#     n = 101
#     ns = np.arange(-(n - 1) / 2, (n + 1) / 2)

#     sinc_window = sinc(ns)
#     kaiser_window = kaiser(n, beta=8.6)
#     window = kaiser_window * sinc_window
#     window = window / np.sum(window)

#     y = np.zeros((len(data) * int(L)) // M, dtype=np.float32)

#     for n in range(y.shape[0]):
#         y_n = 0
#         for k in range(len(window)):
#             h = window[k]
#             m = n * M - k
#             if (m / L).is_integer():
#                 x_m = data[m // L]
#             else:
#                 x_m = 0
#             y_n += x_m * h
#         y[n] = y_n

#     return y


def resample_decimate(data, sr, rec_sr):
    gcd = np.gcd(sr, rec_sr)
    L = sr // gcd
    M = rec_sr // gcd
    return resample_poly(data, L, M, window=("kaiser", 8.6))


def demodulate(y, sr):
    demodulated = np.zeros(y.shape[0])

    phi = 2 * np.pi * f_carrier / sr
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    previous = y[0]

    for i in range(1, y.shape[0]):
        current = y[i]
        # using square of amplitude to avoid sqrt
        demodulated[i] = (
            np.sqrt(previous**2 + current**2 - (previous * current * 2 * cosphi))
            / sinphi
        )
        previous = current

    return demodulated


def decode(filename):
    rec_sr, data = wavfile.read(f"input/{filename}")
    if data.ndim > 1:
        data = data[:, 1]
    sr = 20800
    data = data.astype(float)
    data = data - np.mean(data)

    decimated = resample_decimate(data, sr, rec_sr)

    demodulated = demodulate(decimated, sr)
    signal = resample_decimate(demodulated, f_pixel, sr)
    signal = signal - np.min(signal)
    signal = signal / np.max(np.abs(signal)) * 2 - 1

    sync_frames = get_sync_frames(False)
    correlated = correlate(signal, sync_frames[0], mode="same")
    diff_correlated = -np.diff(correlated)

    inc = 2080
    lines = []
    for ind in range(0, len(signal), inc):
        start = np.argmax(diff_correlated[ind : ind + inc]) + ind - 39 // 2
        line = signal[start : start + inc]
        if len(line) < inc:
            continue
        brightness = (
            (line - np.min(line)) / (np.max(line) - np.min(line)) * 255
        ).astype(np.uint8)
        lines.append(brightness)

    return lines


def amplitude_modulation(data):
    N = data.shape[-1]
    m = 0.7
    t = np.arange(N) / f_sample
    A = 1
    c = A * np.sin(2 * np.pi * f_carrier * t)
    am_data = (1 + m * data) * c

    return am_data


def encode(filename):
    og_img = cv2.imread(f"input/{filename}")
    gray = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    width = 909 * samples_per_pixel

    img = cv2.resize(gray, (width, int(h / w * width)))
    inverted_img = 255 - img

    img = (img / 255.0) * 2 - 1
    inverted_img = (inverted_img / 255.0) * 2 - 1

    sync_frame0, sync_frame1, _ = get_sync_frames()
    sync_frame0 += [-1 for _ in range(47)]
    sync_frame1 += [-1 for _ in range(47)]
    telemetry = [-1 for _ in range(45)]  # blank

    sync_frame0 = np.repeat(sync_frame0, samples_per_pixel)
    sync_frame1 = np.repeat(sync_frame1, samples_per_pixel)
    telemetry = np.repeat(telemetry, samples_per_pixel)

    signal = []

    for i in range(img.shape[0]):
        line = np.concat(
            [sync_frame0, img[i], telemetry, sync_frame1, inverted_img[i], telemetry]
        )
        signal.append(line)

    signal = np.concat(signal)
    signal = amplitude_modulation(signal)
    signal = signal / np.max(np.abs(signal))

    signal = np.int16(signal * 32767)
    wavfile.write("input/tmp.wav", f_sample, signal)


if __name__ == "__main__":
    encode("penguin.jpg")
    lines = decode("tmp.wav")

    image = np.vstack(lines)
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap="gray", aspect="auto")
    plt.show()
