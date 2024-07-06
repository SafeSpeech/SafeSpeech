import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import torch
import bert_vits2.utils as utils
from bert_vits2.models import (
  SynthesizerTrn,
)
from bert_vits2.text.symbols import symbols
from bert_vits2.text.cleaner import clean_text
from bert_vits2.text import cleaned_text_to_sequence, get_bert
import bert_vits2.commons as commons

import soundfile as sf
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='SpeechGuard_Generate_Dataset')
parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
parser.add_argument('--model', type=str, default='Bert_VITS2', choices='Bert_VITS2', help='the surrogate model')
parser.add_argument('--gpu', type=int, default=-1, help='use which gpu')
parser.add_argument('--mode', type=str, default="NLP",
                    choices=["clean", 'random_noise', 'AdvPoison', 'SEP', 'PTA', 'NLP', 'PeNLP', 'RobustNLP'], help='the mode of the SpeechGuard')
parser.add_argument('--checkpoints-path', type=str, default='checkpoints', help='the storing path of the checkpoints')
parser.add_argument('--perturbation-path', type=str, default='perturbation/', help='the storing path of the generated noise')


def main():
    args = parser.parse_args()
    mode = args.mode
    model_name = args.model
    dataset_name = args.dataset
    ckpt_path = args.checkpoints_path
    if args.gpu >= 0:
        device = torch.device(f"cuda:{str(args.gpu)}" if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    config_path = f'bert_vits2/configs/{dataset_name.lower()}_bert_vits2.json'
    hps = utils.get_hparams_from_file(config_path=config_path)

    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6

    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).to(device)
    model.eval()

    output_path = f'./data/synthesized/{dataset_name}/{mode}'
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    if dataset_name == "LibriTTS":
        epoch = 100
    else:
        epoch = 200

    if mode == "FT":
        checkpoint_path = f"{ckpt_path}/{dataset_name}/FT/G_{model_name}_FT_{dataset_name}_{epoch}.pth"
    else:
        checkpoint_path = f"{ckpt_path}/{dataset_name}/G_{model_name}_{mode}_{dataset_name}_{epoch}.pth"

    print(f"Using the checkpoints: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint)
    except:
        model.load_state_dict(checkpoint['model'])
    print(f"checkpoint path is {checkpoint_path}")

    test_txt_path = hps.data.test_files
    with open(test_txt_path, 'r') as f:
        lines = f.readlines()

    for index, line in tqdm(enumerate(lines), total=len(lines)):
        if dataset_name == "LibriTTS":
            audio_path, sid, text = line.split('|')
            output_audio_name = sid + "_" + audio_path.split('/')[3] + "_" + str(index) + '.wav'
            language = "EN"
        else:
            audio_path, sid, text = line.split('|')
            language = "EN"
            output_audio_name = sid + "_" + audio_path.split('/')[3].split('_')[2] + "_" + str(index) + '.wav'

        bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(text, language, hps, device)

        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        speakers = torch.tensor([int(sid)]).to(device)

        noise_scale = 0.2
        noise_scale_w = 0.9
        sdp_ratio = 0.2
        length_scale = 1.0

        audio = model.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert, en_bert,
                            sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                            length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

        rate = hps.data.sampling_rate
        if dataset_name == "LibriTTS":
            assert rate == 24000
        else:
            assert rate == 16000
        output_file_name = os.path.join(output_path, output_audio_name)

        sf.write(output_file_name, audio, samplerate=rate)

    # Get eval lists
    eval_list_path = './evaluation/evallists'
    if os.path.exists(eval_list_path) is False:
        os.mkdir(eval_list_path)
    eval_list = f'./evaluation/evallists/{model_name}_{mode}_{dataset_name}_text.txt'

    syn_path = f'./data/synthesized/{dataset_name}/{mode}'
    gt_audio_path = test_txt_path

    with open(gt_audio_path, 'r') as f:
        gt_audio = f.readlines()
    syn_audio_list = os.listdir(syn_path)
    assert len(syn_audio_list) == len(gt_audio), print(len(syn_audio_list), len(gt_audio))

    with open(eval_list, 'w') as f:
        for index, gt in enumerate(gt_audio):
            gt_path = gt.split('|')[0]
            text = gt.split('|')[2].replace('\n', '')
            if dataset_name == "LibriTTS":
                speaker_id = gt_path.split('/')[3]
            else:
                speaker_id = gt_path.split('/')[3].split('_')[2]

            for syn_audio_path in syn_audio_list:
                syn_audio_name = syn_audio_path[:-4]
                inner_sid = syn_audio_name.split('_')[1]
                inner_index = syn_audio_name.split('_')[2]

                if inner_index == str(index):
                    assert inner_sid == speaker_id
                    gt_write_in = gt_path + '|' + text + '\n'
                    syn_write_in = os.path.join(syn_path, syn_audio_path) + '|' + text + '\n'
                    write_in = gt_write_in + syn_write_in
                    f.write(write_in)
                    break


def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    style_text = None if style_text == "" else style_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert_ori
        ja_bert = torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "JP":
        bert = torch.randn(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "EN":
        bert = torch.randn(1024, len(phone))
        ja_bert = torch.randn(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


if __name__ == "__main__":
    main()