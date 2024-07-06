from bert_vits2.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    WavLMDiscriminator,
)
from bert_vits2.mel_processing import spectrogram_torch, mel_spectrogram_torch
from bert_vits2.text.symbols import symbols
import torch
from torch.nn import functional as F

from bert_vits2.models_N import (
    SynthesizerTrn_N,
    MultiPeriodDiscriminator_N,
    DurationDiscriminator_N
)
from torch_stoi import NegSTOILoss


def compute_l1_distance(hps, wav, wav_hat):
    wav_mel = mel_spectrogram_torch(
        wav.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    wav_hat_mel = mel_spectrogram_torch(
        wav_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    loss_mel_wav = F.l1_loss(wav_mel, wav_hat_mel) * hps.train.c_mel

    return loss_mel_wav


def get_spec(waves, waves_len, hps):
    spec_np = []
    spec_lengths = torch.LongTensor(len(waves))

    device = waves.device
    for index, wave in enumerate(waves):
        audio_norm = wave[:, :waves_len[index]]
        spec = spectrogram_torch(audio_norm,
                                 hps.filter_length, hps.sampling_rate,
                                 hps.hop_length, hps.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        spec_np.append(spec)
        spec_lengths[index] = spec.size(1)

    max_spec_len = max(spec_lengths)
    spec_padded = torch.FloatTensor(len(waves), spec_np[0].size(0), max_spec_len)
    spec_padded.zero_()

    for i, spec in enumerate(waves):
        spec_padded[i][:, :spec_lengths[i]] = spec_np[i]

    return spec_padded.to(device), spec_lengths.to(device)


def build_models(hps, device=torch.device("cpu")):
    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6

    net_dur_disc = DurationDiscriminator(
        hps.model.hidden_channels,
        hps.model.hidden_channels,
        3,
        0.1,
        # gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        gin_channels=hps.model.gin_channels
    ).to(device)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    net_wd = WavLMDiscriminator(
        hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
    ).to(device)

    return net_g, net_d, net_dur_disc, net_wd


def build_models_N(hps, device):
    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6

    net_dur_disc = DurationDiscriminator_N(
        hps.model.hidden_channels,
        hps.model.hidden_channels,
        3,
        0.1,
        gin_channels=hps.model.gin_channels,
    ).to(device)
    net_g = SynthesizerTrn_N(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator_N(hps.model.use_spectral_norm).to(device)
    net_wd = WavLMDiscriminator(
        hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
    ).to(device)

    return net_g, net_d, net_dur_disc, net_wd


def build_optims(hps, models):
    net_g, net_d, net_dur_disc, net_wd = models

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_wd = torch.optim.AdamW(
        net_wd.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    optim_dur_disc = torch.optim.AdamW(
        net_dur_disc.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    return optim_g, optim_d, optim_dur_disc, optim_wd


def build_schedulers(hps, optims, epoch_str=1):
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_wd = torch.optim.lr_scheduler.ExponentialLR(
        optim_wd, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
        optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    return scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd


def compute_stoi(sample_rate, waveforms, perturb_waveforms):
    device = waveforms.device
    stoi_function = NegSTOILoss(sample_rate=sample_rate).to(device)

    loss_stoi = stoi_function(waveforms, perturb_waveforms).mean()
    return loss_stoi


def compute_stft(waveforms, perturb_waveforms):
    stft_clean = torch.stft(waveforms, n_fft=2048, win_length=2048, hop_length=512, return_complex=False)
    stft_p = torch.stft(perturb_waveforms, n_fft=2048, win_length=2048, hop_length=512, return_complex=False)
    loss_stft = torch.norm(stft_p - stft_clean, p=2)

    return loss_stft


def compute_perceptual_loss(hps, p_wav, wav):
    loss_stoi = compute_stoi(hps.data.sampling_rate, wav, p_wav)
    loss_stft = compute_stft(wav.squeeze(1), p_wav.squeeze(1))
    loss_perceptual = loss_stoi + loss_stft

    return loss_perceptual


def get_model_output(model, text, text_len, p_spec, spec_len, speakers, tone, language, bert, ja_bert, en_bert):
    wav_hat, _, _, _, _, _, (_, _, _, _, _, _), (_, _, _, _), _, \
        = model(text, text_len, p_spec, spec_len, speakers, tone, language, bert, ja_bert, en_bert, is_clip=False)

    return wav_hat
