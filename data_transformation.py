import librosa
import random

import torch
import torchaudio
import numpy as np

from tqdm import tqdm

import torch.nn.functional as F
from pysndfx import AudioEffectsChain


def down_up(wav, sampling_rate=24000):
    device  = wav.device
    sample_rate_list = [8000, 12000, 10000]
    target_sr = sample_rate_list[random.randint(0, 2)]

    down_sampled = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr).to(device)(wav)
    up_sampled = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=sampling_rate).to(device)(down_sampled)

    return up_sampled


def wav_to_mel_spectrogram(wav, sampling_rate=24000):
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )
    return frames.astype(np.float32).T


def mel_spectrogram_to_wav(mel_spectrogram, sampling_rate=24000):
    mel_spectrogram = mel_spectrogram.T
    reconstructed_wav = librosa.feature.inverse.mel_to_audio(mel_spectrogram,
                                                             n_fft=2048,
                                                             hop_length=512,
                                                             sr=sampling_rate)

    return reconstructed_wav


def mel_inverse(wav, sampling_rate=24000):
    mel_spectrogram = wav_to_mel_spectrogram(wav, sampling_rate)
    inverse_wav = mel_spectrogram_to_wav(mel_spectrogram, sampling_rate)

    return inverse_wav


# def change_speed(wav):
#     speed_list = [0.8, 0.9, 1.0, 1.1, 1.2]
#     speed = speed_list[random.randint(0, 4)]
#     wav_changed = librosa.effects.time_stretch(wav, rate=speed)
#
#     return wav_changed


def change_speed(wav):
    speed_list = [0.8, 0.9, 1.0, 1.1, 1.2]
    speed = speed_list[random.randint(0, 4)]

    wav_changed = librosa.effects.time_stretch(wav, rate=speed)
    wav_inverse = librosa.effects.time_stretch(wav_changed, rate=1 / speed)

    return wav_inverse


def quantize_np(x: np.ndarray, quantize_bits: int) -> np.ndarray:
    return (x + 1.0) * (2 ** quantize_bits - 1) / 2


def dequantize_np(x, quantize_bits) -> np.ndarray:
    return 2 * x / (2 ** quantize_bits - 1) - 1


def quantize(x: torch.Tensor, quantize_bits: int) -> torch.Tensor:
    return (x + 1.0) * (2 ** quantize_bits - 1) / 2


def dequantize(x: torch.Tensor, quantize_bits: int) -> torch.Tensor:
    return 2 * x / (2 ** quantize_bits - 1) - 1


def reduce_noise_power(y, sr=24000):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent)) * 1.5
    threshold_l = round(np.median(cent)) * 0.1

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0,
                                                                                                      frequency=threshold_h,
                                                                                                      slope=0.5)
    y_clean = less_noise(y)

    return y_clean


def quantization(wav, bits=8):
    quantized_audio = quantize(wav, bits)
    dequantized_audio = dequantize(quantized_audio, bits)

    return dequantized_audio


def transformations(mode, wave, sampling_rate):
    if mode == "down_up":
        inverse_wave = down_up(wave, sampling_rate)
    elif mode == "mel_inverse":
        inverse_wave = mel_inverse(wave, sampling_rate)
    elif mode == "quantize":
        inverse_wave = quantization(wave, 8)
    elif mode == "change_speed":
        inverse_wave = change_speed(wave)
    elif mode == "filter":
        inverse_wave = reduce_noise_power(wave, sampling_rate)
    elif mode == "None":
        inverse_wave = wave
    else:
        raise Exception

    return inverse_wave


def transform_batch(waves, mode, sampling_rate):
    device = waves.device
    wave_list = [None] * waves.size(0)
    if mode == "down_up":
        return down_up(waves, 24000)
    if mode == "quantize":
        return quantization(waves, 8)
    for index in tqdm(range(len(waves))):
        wave = waves[index].squeeze(0)
        wave_np = transformations(mode, wave.cpu().numpy(), sampling_rate)
        wave_list[index] = torch.from_numpy(wave_np).unsqueeze(0).to(device)

    new_wave = torch.stack(wave_list, dim=0)

    return new_wave
