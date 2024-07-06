import argparse
import torch
import os
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import time
from tqdm import tqdm

import bert_vits2.utils
from bert_vits2.data_utils import (
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader
)
from bert_vits2.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from bert_vits2.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    WavLMLoss
)
import bert_vits2.commons as commons

from toolbox import (
    build_models,
    build_optims,
    build_schedulers,
    get_spec
)


parser = argparse.ArgumentParser(description='SpeechGuard_Protected_Train')
parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
parser.add_argument('--model', type=str, default='Bert_VITS2', choices='Bert_VITS2', help='the surrogate model')
parser.add_argument('--batch-size', type=int, default=2, help='the batch size of protected and training')
parser.add_argument('--gpu', type=int, default=-1, help='use which gpu')
parser.add_argument('--random-seed', type=int, default=1234, help='random seed')
parser.add_argument('--protection-mode', type=str, default="NLP",
                    choices=['random_noise', 'AdvPoison', 'SEP', 'PTA', 'NLP', 'PeNLP', 'RobustNLP'], help='the protection mode of the SpeechGuard')
parser.add_argument('--checkpoints-path', type=str, default='checkpoints', help='the storing path of the checkpoints')
parser.add_argument('--perturbation-path', type=str, default='perturbation/', help='the storing path of the generated noise')
parser.add_argument('--epsilon', type=int, default=8, help='the perturbation radius boundary')
parser.add_argument('--perturbation-epochs', type=int, default=2, help='the iteration numbers of the proptecte perturbation')


def main():
    args = parser.parse_args()
    mode = args.protection_mode
    model_name = args.model
    dataset_name = args.dataset
    batch_size = args.batch_size
    if args.gpu >= 0:
        device = torch.device(f"cuda:{str(args.gpu)}" if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    ckpt_path = args.checkpoints_path
    output_ckpt_path = f'{ckpt_path}/{dataset_name}'
    if os.path.exists(output_ckpt_path) is not True:
        os.mkdir(output_ckpt_path)

    config_path = f'bert_vits2/configs/{dataset_name.lower()}_bert_vits2.json'
    hps = bert_vits2.utils.get_hparams_from_file(config_path=config_path)
    hps.train.batch_size = batch_size

    seed = args.random_seed
    torch.manual_seed(seed)
    hps.data.training_files = 'bert_vits2/filelists/test.txt'
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset,
                              num_workers=4,
                              shuffle=False,
                              collate_fn=collate_fn,
                              batch_size=hps.train.batch_size,
                              pin_memory=True,
                              drop_last=False)

    if dataset_name == "LibriTTS":
        assert len(train_dataset) == 2, print("Please ensure that the number of LibriTTS is 108 for training!")
    elif dataset_name == "CMU_ARCTIC":
        assert len(train_dataset) == 1440, print("Please ensure that the number of LibriTTS is 1440 for training!")

    checkpoints_path = args.checkpoints_path
    hps.model_dir = os.path.join(checkpoints_path, 'base_models')
    assert os.path.exists(hps.model_dir), print("The move the pre-trained checkpoints to 'checkpoints/base_models'")
    assert len(os.listdir(hps.model_dir)) == 4, print("There needs four checkpoints.")

    models = build_models(hps, device)
    net_g, net_d, net_dur_disc, net_wd = models
    optims = build_optims(hps, models)
    optim_g, optim_d, optim_dur_disc, optim_wd = optims

    dur_resume_lr = hps.train.learning_rate
    wd_resume_lr = hps.train.learning_rate

    _, _, dur_resume_lr, epoch_str = bert_vits2.utils.load_checkpoint(
        bert_vits2.utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
        net_dur_disc,
        optim_dur_disc,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_dur_disc.param_groups[0].get("initial_lr"):
        optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

    _, optim_g, g_resume_lr, epoch_str = bert_vits2.utils.load_checkpoint(
        bert_vits2.utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
        net_g,
        optim_g,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )

    _, optim_d, d_resume_lr, epoch_str = bert_vits2.utils.load_checkpoint(
        bert_vits2.utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
        net_d,
        optim_d,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_g.param_groups[0].get("initial_lr"):
        optim_g.param_groups[0]["initial_lr"] = g_resume_lr
    if not optim_d.param_groups[0].get("initial_lr"):
        optim_d.param_groups[0]["initial_lr"] = d_resume_lr
        # optim_d.param_groups[0]["initial_lr"] = 0.0002

    _, optim_wd, wd_resume_lr, epoch_str = bert_vits2.utils.load_checkpoint(
        bert_vits2.utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth"),
        net_wd,
        optim_wd,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_wd.param_groups[0].get("initial_lr"):
        optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr

    epoch_str = max(epoch_str, 1)
    global_step = int(bert_vits2.utils.get_steps(bert_vits2.utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")))
    print(f"******************检测到模型存在，epoch为 {epoch_str}，gloabl step为 {global_step}*********************")

    schedulers = build_schedulers(hps, optims, epoch_str)
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers

    scaler = GradScaler(enabled=hps.train.bf16_run)

    wl = WavLMLoss(
        hps.model.slm.model,
        net_wd,
        hps.data.sampling_rate,
        hps.model.slm.sr,
    ).to(device)

    perturbation_path = args.perturbation_path
    perturb_path = f'{perturbation_path}/{model_name}_{mode}_{dataset_name}.noise'
    assert os.path.exists(perturb_path)
    perturbations = torch.load(perturb_path, map_location="cpu")

    training_epochs = 10
    print(f"Begin to training, and it will train for {training_epochs} epochs!")

    start_time = time.time()
    for epoch in range(1, training_epochs + 1):
        loss_gen_list, loss_disc = perturb_train(
            epoch,
            hps,
            [net_g, net_d, net_dur_disc, net_wd, wl],
            [optim_g, optim_d, optim_dur_disc, optim_wd],
            [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
            scaler,
            train_loader,
            device,
            perturbations
        )

        loss_gen, loss_fm, loss_kl, loss_mel = loss_gen_list

        scheduler_g.step()
        scheduler_d.step()
        scheduler_wd.step()
        scheduler_dur_disc.step()

        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
        print(
            f"[{formatted_time}] Epoch {epoch}: D loss {loss_disc:.4f}, loss gen {loss_gen:.4f}, fm {loss_fm:.4f}, "
            f"mel {loss_mel:.4f}, kl {loss_kl:.4f}")
        # print(f"[{formatted_time}] Epoch {epoch}: D loss {loss_disc:.4f}, G loss {loss_gen_list}")

        if epoch % 20 == 0:
            store_path = f"./{checkpoints_path}/{dataset_name}/G_{model_name}_{mode}_{dataset_name}_{epoch}.pth"
            torch.save(net_g.state_dict(), store_path)

    store_path = f"./{checkpoints_path}/{dataset_name}/G_{model_name}_{mode}_{dataset_name}_{epoch}.pth"
    torch.save(net_g.state_dict(), store_path)


def perturb_train(epoch, hps, nets, optims, schedulers, scaler, loader, device, perturbations):
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers

    net_g.train()
    net_d.train()
    net_wd.train()
    net_dur_disc.train()

    global_step = (epoch - 1) * len(loader)

    # for batch_index, batch in tqdm(enumerate(loader), total=len(loader)):
    for batch_index, batch in enumerate(loader):
        text, text_len, _, _, wav, wav_len, speakers, \
        tone, language, bert, ja_bert, en_bert = batch

        if net_g.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.mas_noise_scale_initial
                - net_g.noise_scale_delta * global_step
            )
            net_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        text, text_len = text.to(device, non_blocking=True), text_len.to(device, non_blocking=True)
        # spec, spec_len = spec.to(device, non_blocking=True), spec_len.to(device, non_blocking=True)
        wav, wav_len = wav.to(device, non_blocking=True), wav_len.to(device, non_blocking=True)
        speakers, tone, language = speakers.to(device, non_blocking=True), tone.to(device,non_blocking=True), language.to(device, non_blocking=True)
        bert, ja_bert, en_bert = bert.to(device, non_blocking=True), ja_bert.to(device, non_blocking=True), en_bert.to(device, non_blocking=True)
        perturbation = perturbations[batch_index].to(device, non_blocking=True)

        p_wav = torch.add(wav.data, perturbation)
        p_wav = torch.clamp(p_wav, min=-1., max=1.)

        p_spec, spec_len = get_spec(p_wav, wav_len, hps.data)

        wav_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), \
            (hidden_x, logw, logw_, logw_sdp), g, \
            = net_g(
            text, text_len, p_spec, spec_len, speakers, tone, language, bert, ja_bert, en_bert,
        )

        mel = spec_to_mel_torch(
            p_spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        wav_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        wav_hat_mel = mel_spectrogram_torch(
            wav_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        wav = commons.slice_segments(wav, ids_slice * hps.data.hop_length, hps.train.segment_size)

        # Discriminator
        wav_d_hat_r, wav_d_hat_g, _, _ = net_d(wav, wav_hat.detach())

        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(wav_d_hat_r, wav_d_hat_g)
        loss_disc_all = loss_disc

        if net_dur_disc is not None:
            wav_dur_hat_r, wav_dur_hat_g = net_dur_disc(
                hidden_x.detach(),
                x_mask.detach(),
                logw_.detach(),
                logw.detach(),
                g.detach(),
            )
            wav_dur_hat_r_sdp, wav_dur_hat_g_sdp = net_dur_disc(
                hidden_x.detach(),
                x_mask.detach(),
                logw_.detach(),
                logw_sdp.detach(),
                g.detach(),
            )
            wav_dur_hat_r = wav_dur_hat_r + wav_dur_hat_r_sdp
            wav_dur_hat_g = wav_dur_hat_g + wav_dur_hat_g_sdp

            loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(wav_dur_hat_r, wav_dur_hat_g)
            loss_dur_disc_all = loss_dur_disc

            optim_dur_disc.zero_grad()
            loss_dur_disc_all.backward()
            grad_norm_dur = commons.clip_grad_value_(net_dur_disc.parameters(), None)
            optim_dur_disc.step()

        optim_d.zero_grad()
        loss_disc_all.backward()
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        global_step += 1

        loss_slm = wl.discriminator(wav.detach().squeeze(), wav_hat.detach().squeeze()).mean()

        optim_wd.zero_grad()
        loss_slm.backward()
        grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
        optim_wd.step()

        # Generator
        _, wav_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
        _, wav_dur_hat_g_sdp = net_dur_disc(hidden_x, x_mask, logw_, logw_sdp, g)
        wav_dur_hat_g = wav_dur_hat_g + wav_dur_hat_g_sdp

        wav_d_hat_r, wav_d_hat_g, fmap_r, fmap_g = net_d(wav, wav_hat)

        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(wav_mel, wav_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(wav_d_hat_g)

        loss_lm = wl(wav.detach().squeeze(), wav_hat.squeeze()).mean()
        loss_lm_gen = wl.generator(wav_hat.squeeze())

        loss_dur_gen, losses_dur_gen = generator_loss(wav_dur_hat_g)

        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_lm + loss_lm_gen + loss_dur_gen

        optim_g.zero_grad()
        loss_gen_all.backward()
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        optim_g.step()

    return [loss_gen.item(), loss_fm.item(), loss_kl.item(), loss_mel.item()], loss_disc_all.item()


if __name__ == "__main__":
    main()
