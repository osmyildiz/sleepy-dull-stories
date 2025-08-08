import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import soundfile as sf
import os

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_high(x, fs, cutoff=6000.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='high')
    return lfilter(b, a, x)

def smooth_envelope(x, fs, attack_ms=5.0, release_ms=50.0):
    attack = max(1, int(fs * attack_ms/1000.0))
    release = max(1, int(fs * release_ms/1000.0))
    env = np.zeros_like(x)
    prev = 0.0
    for i, val in enumerate(np.abs(x)):
        coeff = np.exp(-1.0/(attack if val > prev else release))
        prev = coeff*prev + (1-coeff)*val
        env[i] = prev
    return env

def dynamic_de_ess(x, fs, band=(4000.0, 9000.0), threshold_db=-28.0, ratio=6.0, attack_ms=3.0, release_ms=60.0):
    b, a = butter_bandpass(band[0], band[1], fs, order=4)
    sib = lfilter(b, a, x)
    base = x - sib
    env = smooth_envelope(sib, fs, attack_ms=attack_ms, release_ms=release_ms)
    eps = 1e-9
    env_db = 20*np.log10(env + eps)
    over = env_db - threshold_db
    gain_db = np.where(over > 0, -(1 - 1/ratio) * over, 0.0)
    gain = 10**(gain_db/20.0)
    sib_comp = sib * gain
    return base + sib_comp

def gentle_high_tilt(x, fs, cutoff=6000.0, db=-2.5):
    hp = butter_high(x, fs, cutoff=cutoff, order=2)
    g = 10**(db/20.0)
    return x + g*hp

def limiter_peak(x, ceiling_db=-1.0):
    peak = np.max(np.abs(x)) + 1e-12
    ceiling = 10**(ceiling_db/20.0)
    if peak > ceiling:
        x = x * (ceiling/peak)
    return x

def process_audio(input_path):
    audio = AudioSegment.from_file(input_path)
    sr = audio.frame_rate
    mono = audio.set_channels(1)
    samples = np.array(mono.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(np.int16).max

    y = dynamic_de_ess(samples, sr)
    y = gentle_high_tilt(y, sr, db=-2.5)
    y = limiter_peak(y, ceiling_db=-1.0)

    output_path = os.path.splitext(input_path)[0] + "_processed.wav"
    sf.write(output_path, y, sr)
    print(f"Processed file saved as: {output_path}")

# KULLANIM
process_audio("/Users/nilgun/Downloads/scene_16_2.mp3")  # Buraya kendi dosya adını yaz
