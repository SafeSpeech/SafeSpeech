import argparse
import torch
import os
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import time
from torch.autograd import Variable
from tqdm import tqdm
from torch.nn import functional as F
import random

import bert_vits2.utils
from bert_vits2.data_utils import (
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader
)
from bert_vits2.mel_processing import mel_spectrogram_torch
from bert_vits2.losses import WavLMLoss
import bert_vits2.commons as commons

from toolbox import (
    build_models,
    build_models_N,
    build_optims,
    build_schedulers,
    compute_l1_distance,
    compute_perceptual_loss,
    get_spec,
    get_model_output
)
from data_transformation import transform_batch

parser = argparse.ArgumentParser(description='SpeechGuard_Protection')
parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
parser.add_argument('--model', type=str, default='Bert_VITS2', choices='Bert_VITS2', help='the surrogate model')
parser.add_argument('--batch-size', type=int, default=27, help='the batch size of protected and training')
parser.add_argument('--gpu', type=int, default=0, help='use which gpu')
parser.add_argument('--random-seed', type=int, default=1234, help='random seed')
parser.add_argument('--protection-mode', type=str, default="RobustNLP",
                    choices=['random_noise', 'AdvPoison', 'SEP', 'PTA', 'NLP', 'PeNLP', 'RobustNLP'], help='the protection mode of the SpeechGuard')
parser.add_argument('--checkpoints-path', type=str, default='checkpoints', help='the storing path of the checkpoints')
parser.add_argument('--perturbation-path', type=str, default='perturbation/', help='the storing path of the generated perturbation')
parser.add_argument('--epsilon', type=int, default=8, help='the perturbation radius boundary')
parser.add_argument('--perturbation-epochs', type=int, default=200, help='the iteration numbers of the noise')


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

    config_path = f'bert_vits2/configs/{dataset_name.lower()}_bert_vits2.json'
    hps = bert_vits2.utils.get_hparams_from_file(config_path=config_path)
    hps.train.batch_size = batch_size

    seed = args.random_seed
    torch.manual_seed(seed)
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
        assert len(train_dataset) == 108, print("Please ensure that the number of LibriTTS is 108 for training!")
    elif dataset_name == "CMU_ARCTIC":
        assert len(train_dataset) == 1440, print("Please ensure that the number of LibriTTS is 1440 for training!")

    checkpoints_path = args.checkpoints_path
    hps.model_dir = os.path.join(checkpoints_path, 'base_models')
    assert os.path.exists(hps.model_dir), print("The move the pre-trained checkpoints to 'checkpoints/base_models'")
    assert len(os.listdir(hps.model_dir)) == 4, print("There needs four checkpoints.")

    models = build_models_N(hps, device)
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
    print(f"******************The epoch is {epoch_str}，global step is {global_step}*********************")

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
    if os.path.exists(perturbation_path) is not True:
        os.mkdir(perturbation_path)

    noises = [None] * len(train_loader)
    epsilon = args.epsilon / 255
    alpha = epsilon / 10
    max_epoch = args.perturbation_epochs

    for param in net_g.parameters():
        param.requires_grad = False
    for param in net_d.parameters():
        param.requires_grad = False
    for param in net_dur_disc.parameters():
        param.requires_grad = False
    for param in net_wd.parameters():
        param.requires_grad = False

    if mode == "SEP":
        generator_path = f'./{checkpoints_path}/base_models/G_0.pth'
        checkpoints_path = f'./{checkpoints_path}/{dataset_name}/FT/'
        checkpoints_list = os.listdir(checkpoints_path)
        G_checkpoints = [file for file in checkpoints_list if file.startswith("G")]
        G_checkpoints = [os.path.join(checkpoints_path, file) for file in G_checkpoints]
        G_checkpoints.append(generator_path)

        assert len(G_checkpoints) == 11, print("Please store 10 intermedia checkpoints in training for SEP mode.")
    else:
        G_checkpoints = None

    print(f"The intermediate checkpoint lists: {G_checkpoints}")

    print(f"The protection method is {mode}, and batch length is {len(train_loader)}")
    start_time = time.time()
    for batch_index, batch in enumerate(train_loader):
        # if batch_index == 2:
        #     continue
        # if batch_index < max_index:
        #     continue
        noises[batch_index], loss = error_min(
            hps, [net_g, net_d, net_dur_disc, net_wd, wl], batch,
            epsilon, alpha, max_epoch, noises[batch_index], mode, seed,
            checkpoints=G_checkpoints
        )
        torch.cuda.empty_cache()

        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
        print(f'[{formatted_time}] Batch {batch_index}, the loss is {loss:.6f}')

        # Prevent long time running
        if (batch_index + 1) % 5 == 0:
            torch.save(noises,
                       f'./{perturbation_path}/{model_name}_{mode}_{dataset_name}_{batch_index + 1}.noise')

    store_path = f'./{perturbation_path}/{model_name}_{mode}_{dataset_name}.noise'
    torch.save(noises, store_path)


def error_min(hps, nets, batch_batch, epsilon, alpha, max_epoch, noise, mode, seed, checkpoints=None):
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    device = next(net_g.parameters()).device
    text, text_len, spec, spec_len, wav, wav_len, speakers, \
        tone, language, bert, ja_bert, en_bert = batch_batch
    text, text_len = text.to(device), text_len.to(device)
    wav, wav_len = wav.to(device), wav_len.to(device)
    speakers, tone, language = speakers.to(device), tone.to(device), language.to(device)
    bert, ja_bert, en_bert = bert.to(device), ja_bert.to(device), en_bert.to(device)

    if mode == "random_noise":
        random_noise = torch.randn(wav.shape).to(device)
        max_item = random_noise.abs().max().item()
        noise_norm = random_noise / max_item * epsilon

        return noise_norm, 0.0

    # if noise is None:
    noise = torch.zeros(wav.shape).to(device)

    ori_wav = wav
    bound_max, bound_min = ori_wav.max(), ori_wav.min()

    p_wav = Variable(ori_wav.data + noise, requires_grad=True)
    p_wav = Variable(torch.clamp(p_wav, min=bound_min, max=bound_max), requires_grad=True)

    opt_noise = torch.optim.SGD([p_wav], lr=5e-2)

    net_g.train()
    # for iteration in tqdm(range(max_epoch)):
    for iteration in range(max_epoch):
        opt_noise.zero_grad()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        p_spec, spec_len = get_spec(p_wav, wav_len, hps.data)

        if mode == "NLP":
            weight = 10

            wav_hat = get_model_output(net_g, text, text_len, p_spec, spec_len, speakers,
                                       tone, language, bert, ja_bert, en_bert)
            if iteration == 0:
                random_noise = torch.randn(wav_hat.shape).to(device)

            loss_mel = compute_l1_distance(hps, p_wav, wav_hat)

            # Noise-Leading Protection
            loss_nl = compute_kl_divergence(hps, wav_hat, random_noise)
            loss_nm = compute_l1_distance(hps, wav_hat, random_noise)

            loss = loss_mel + weight * (loss_nl + loss_nm)

            # grad = torch.autograd.grad(loss, p_wav)[0]
            loss.backward()
            grad = p_wav.grad

        elif mode == "PeNLP":
            weight = 10
            perception_weight = 0.05

            wav_hat = get_model_output(net_g, text, text_len, p_spec, spec_len, speakers,
                                       tone, language, bert, ja_bert, en_bert)
            if iteration == 0:
                random_noise = torch.randn(wav_hat.shape).to(device)

            loss_mel = compute_l1_distance(hps, p_wav, wav_hat)

            loss_nl = compute_kl_divergence(hps, wav_hat, random_noise)
            loss_nm = compute_l1_distance(hps, wav_hat, random_noise)

            loss_perceptual = compute_perceptual_loss(hps, p_wav, wav)
            loss = loss_mel + weight * (loss_nl + loss_nm) + perception_weight * loss_perceptual

            # grad = torch.autograd.grad(loss, p_wav)[0]
            loss.backward()
            grad = p_wav.grad

        elif mode == "AdvPoison":
            wav_hat = get_model_output(net_g, text, text_len, p_spec, spec_len, speakers,
                                       tone, language, bert, ja_bert, en_bert)

            loss_mel = compute_l1_distance(hps, p_wav, wav_hat)
            loss = loss_mel
            loss.backward()
            grad = p_wav.grad * -1.

        elif mode == "PTA":
            batch_len = wav.size(0)
            ids_slice = torch.zeros((batch_len), dtype=torch.long, device=device)
            loss = compute_patch_loss(hps, net_g, text, text_len, p_wav, p_spec, spec_len, speakers,
                                      tone, language, bert, ja_bert, en_bert, device)

            grad = torch.autograd.grad(loss, p_wav)[0]
            noise = alpha * torch.sign(grad) * -1.

            patch_noise = commons.slice_segments(noise, ids_slice * hps.data.hop_length, hps.train.segment_size)
            patch_noise = torch.clamp(patch_noise, min=-epsilon, max=epsilon)
            noise, p_wav = patch_to_all(hps, patch_noise, p_wav.data, wav_len)
            noise = torch.clamp(p_wav.data - ori_wav.data, min=-epsilon, max=epsilon)
            p_wav = Variable(ori_wav.data + noise, requires_grad=True)
            p_wav = Variable(torch.clamp(p_wav, min=bound_min, max=bound_max), requires_grad=True)

            continue
        elif mode == "SEP":
            assert checkpoints is not None

            loss, grad = SEP_Loss(hps, net_g, checkpoints, text, text_len, p_wav, wav_len, p_spec,
                                  spec_len, speakers, tone, language, bert, ja_bert, en_bert, device)

        elif mode == "RobustNLP":
            # segment_number = 10
            # loss = EOT(hps, net_g, text, text_len, p_wav, wav_len, speakers,
            #            tone, language, bert, ja_bert, en_bert, noise, device, number=segment_number)

            # Apply data augmentation when perturbation generation
            p_waves_trans = wav_trans(hps, p_wav)
            p_spec, spec_len = get_spec(p_waves_trans, wav_len, hps.data)
            weight = 10

            wav_hat = get_model_output(net_g, text, text_len, p_spec, spec_len, speakers,
                                       tone, language, bert, ja_bert, en_bert)
            if iteration == 0:
                random_noise = torch.randn(wav_hat.shape).to(device)

            loss_mel = compute_l1_distance(hps, p_waves_trans, wav_hat)

            loss_nl = compute_kl_divergence(hps, wav_hat, random_noise)
            loss_nm = compute_l1_distance(hps, wav_hat, random_noise)

            loss = loss_mel + weight * (loss_nl + loss_nm)

            # grad = torch.autograd.grad(loss, p_wav)[0]
            loss.backward()
            grad = p_wav.grad

        else:
            raise Exception

        noise = alpha * torch.sign(grad) * -1.
        p_wav = Variable(p_wav.data + noise, requires_grad=True)
        noise = torch.clamp(p_wav.data - ori_wav.data, min=-epsilon, max=epsilon)
        p_wav = Variable(ori_wav.data + noise, requires_grad=True)
        p_wav = Variable(torch.clamp(p_wav, min=bound_min, max=bound_max), requires_grad=True)


    try:
        return noise, loss.item()
    except:
        return noise, loss


def compute_kl_divergence(hps, x_hat, z):
    x_mel = mel_spectrogram_torch(
        x_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    z_mel = mel_spectrogram_torch(
        z.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )

    p_log = F.log_softmax(x_mel, dim=-1)
    q = F.softmax(z_mel, dim=-1)

    # p_log = F.log_softmax(x_hat, dim=-1)
    # q = F.softmax(z, dim=-1)

    kl_divergence = F.kl_div(p_log, q, reduction="batchmean")

    return kl_divergence


def compute_patch_loss(hps, net_g, text, text_len, p_wav, p_spec, spec_len, speakers,
                       tone, language, bert, ja_bert, en_bert, device):
    batch_size = text.size(0)
    ids_str = torch.zeros((batch_size), dtype=torch.long).to(device)
    wav_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), \
        (hidden_x, logw, logw_, logw_sdp), g \
        = net_g(text, text_len, p_spec, spec_len, speakers, tone, language, bert, ja_bert, en_bert, is_clip=True, ids_str=ids_str)
    p_wav_slice = commons.slice_segments(p_wav, ids_slice * hps.data.hop_length, hps.train.segment_size)

    loss = compute_l1_distance(hps, p_wav_slice, wav_hat)

    return loss


def patch_to_all(hps, noises, waveforms, lengths):
    device = waveforms.device
    # the shape of the perturbation is [batch_size, 1, segment_size] for PTA
    segment_size = hps.train.segment_size
    noises_new = torch.zeros_like(waveforms)

    for index, wave in enumerate(waveforms):
        length = lengths[index]
        noise = noises[index]
        patch_batch = length // segment_size
        traversed_noise = torch.zeros((1, wave.size(1))).to(device)
        patch_noise = noise.repeat(1, patch_batch)

        start_index = patch_batch * segment_size
        traversed_noise[:, :start_index] = patch_noise
        end_index = length - start_index
        traversed_noise[:, start_index: length] = noise[:, : end_index]
        wave += traversed_noise
        noises_new[index] = traversed_noise

    return noises_new, waveforms


def SEP(hps, model, text, text_len, p_wav, wav_len, p_spec,
        spec_len, speakers, tone, language, bert, ja_bert, en_bert):

    p_wav.requires_grad = True
    p_spec, spec_len = get_spec(p_wav, wav_len, hps.data)

    wav_hat = get_model_output(model, text, text_len, p_spec, spec_len, speakers,
                               tone, language, bert, ja_bert, en_bert)
    p_wav_slice = p_wav
    loss = compute_l1_distance(hps, p_wav_slice, wav_hat)
    grad = torch.autograd.grad(loss, p_wav)[0]

    return p_wav.detach(), grad.detach(), loss.item()


def SEP_Loss(hps, model, checkpoints, text, text_len, p_wav, wav_len, p_spec,
             spec_len, speakers, tone, language, bert, ja_bert, en_bert, device):

    grad_list, losses = [], []
    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint)
        except:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            checkpoint_dict = checkpoint['model']
            for layer_name, layer_params in model.state_dict().items():
                if layer_name in checkpoint_dict:
                    checkpoint_dict_param = checkpoint_dict[layer_name]
                    if checkpoint_dict_param.shape == layer_params.shape:
                        model.state_dict()[layer_name].copy_(checkpoint_dict_param)

        # if ckpt_path == './checkpoints/models/G_0.pth':
        #     checkpoint = torch.load(ckpt_path, map_location='cpu')
        #     checkpoint_dict = checkpoint['model']
        #     for layer_name, layer_params in net_g.state_dict().items():
        #         if layer_name in checkpoint_dict:
        #             checkpoint_dict_param = checkpoint_dict[layer_name]
        #             if checkpoint_dict_param.shape == layer_params.shape:
        #                 model.state_dict()[layer_name].copy_(checkpoint_dict_param)
        # else:
        #     checkpoint = torch.load(ckpt_path, map_location="cpu")['model']
        #     model.load_state_dict(checkpoint)

        p_wav, grad, loss = SEP(hps, model, text, text_len, p_wav, wav_len, p_spec,
                                spec_len, speakers, tone, language, bert, ja_bert, en_bert)
        losses.append(loss)
        grad_list.append(grad)

        # print(ckpt_path, loss)

    grad_avg = sum(grad_list) / len(grad_list)
    loss_avg = sum(losses) / len(losses)

    return loss_avg, grad_avg


def EOT(hps, model, text, text_len, p_wav, wav_len, speakers,
            tone, language, bert, ja_bert, en_bert, noise, device, number=10):
    weight = 10
    size_list = [4096, 8192, 12288]
    segment_sizes = random.sample(size_list, 1)

    average_loss = torch.zeros((1)).to(device)
    segment_list = [random.random() for _ in range(number)]

    transform_list = ["down_up", "quantize"]
    trans_index = random.randint(0, 1)

    p_trans = transform_batch(p_wav, mode=transform_list[trans_index], sampling_rate=hps.data.sampling_rate)
    p_spec, spec_len = get_spec(p_trans, wav_len, hps.data)

    for size in segment_sizes:
        segment_size = size // hps.data.hop_length
        for segment_index in segment_list:
            ids_str_max = spec_len - segment_size
            ids_str = (torch.tensor([segment_index]).to(device=p_spec.device) * ids_str_max).to(dtype=torch.long)

            wav_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), \
                (hidden_x, logw, logw_, logw_sdp), g, \
                = model(text, text_len, p_spec, spec_len, speakers, tone, language, bert, ja_bert, en_bert,
                        is_clip=True, ids_str=ids_str, segment_size=segment_size)

            p_wav_slice = commons.slice_segments(p_trans, ids_slice * hps.data.hop_length, segment_size * hps.data.hop_length)
            loss_mel = compute_l1_distance(hps, p_wav_slice, wav_hat)

            noise_slice = commons.slice_segments(noise, ids_slice * hps.data.hop_length, segment_size * hps.data.hop_length)
            loss_nl = compute_kl_divergence(hps, wav_hat, noise_slice)
            loss_nm = compute_l1_distance(hps, wav_hat, noise_slice)

            loss = loss_mel + weight * (loss_nl + loss_nm)

            average_loss += loss

    average_loss /= len(segment_list)
    return average_loss


def wav_trans(hps, wave):
    transform_list = ["down_up", "quantize"]
    trans_index = random.randint(0, 1)

    p_trans = transform_batch(wave, mode=transform_list[trans_index], sampling_rate=hps.data.sampling_rate)

    return p_trans


if __name__ == "__main__":
    main()
