import argparse

import torch.cuda
from pymcd.mcd import Calculate_MCD
from tqdm import tqdm
import whisper
import jiwer
import torchaudio

# If your speechbrain version==1.0.0
from speechbrain.inference.speaker import SpeakerRecognition
# If your speechbrain version<1.0.0
# from speechbrain.pretrained import SpeakerRecognition

parser = argparse.ArgumentParser(description='SpeechGuard_Evaluation')
parser.add_argument('--eval-list', type=str, default='evaluation/evallists/Bert_VITS2_NLP_LibriTTS_text.txt',
                    help='the dataset')


def main():
    args = parser.parse_args()
    eval_list = args.eval_list

    # MCD
    with open(eval_list, 'r') as f:
        audio_list = f.readlines()

    gt_audio_list = []
    syn_audio_list = []
    for index, audio_path in enumerate(audio_list):
        if index % 2 == 0:
            gt_audio_list.append(audio_path)
        else:
            syn_audio_list.append(audio_path)

    results = compute_mcd(gt_audio_list, syn_audio_list)
    print(f"MCD: ", results)

    # WER
    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
    model = whisper.load_model("medium.en", device=device).to(device)

    text_file_path = eval_list
    with open(text_file_path, 'r') as f:
        lines = f.readlines()

    WER_gt, WER_syn = 0.0, 0.0
    for index, line in tqdm(enumerate(lines), total=len(lines)):
        if index % 2 == 0:
            continue
        audio_path, gt_text = line.split('|')
        result = model.transcribe(audio_path, language="en")
        gen_text = result['text']
        wer = jiwer.wer(gt_text, gen_text)

        if index % 2 == 0:
            WER_gt += wer
        else:
            WER_syn += wer

    WER_gt /= (len(lines) // 2)
    WER_syn /= (len(lines) // 2)
    print(f"Syn WER is {WER_syn:.6f}")


    # SIM
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir="encoder/spkrec-ecapa-voxceleb",
                                                   run_opts={"device": device})
    assert len(gt_audio_list) == len(syn_audio_list)
    with torch.no_grad():
        sim = 0.

        for gt_path, syn_path in tqdm(zip(gt_audio_list, syn_audio_list), total=len(gt_audio_list)):
            gt_path, syn_path = gt_path.split('|')[0].replace('\n', ''), syn_path.split('|')[0].replace('\n', '')
            score, prediction = compute_sim(verification, gt_path, syn_path)
            sim += score

        sim = sim / len(gt_audio_list)

        print(f"SIM {sim:.6f}.")


def compute_mcd(gt_list, syn_list):
    torch.cuda.empty_cache()

    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")

    mcd_value = 0.0


    count = 0
    assert len(gt_list) == len(syn_list)
    with torch.no_grad():
        for gt_path, syn_path in tqdm(zip(gt_list, syn_list), total=len(gt_list)):
            try:
                gt_path, syn_path = gt_path.split('|')[0].replace('\n', ''), syn_path.split('|')[0].replace('\n', '')

                # MCD calculation
                mcd = mcd_toolbox.calculate_mcd(gt_path, syn_path)
                mcd_value += mcd

            except Exception as e:
                count += 1

    sum = len(syn_list) - count

    return {
        'MCD': mcd_value / sum
    }


def compute_sim(model, path_1, path_2):
    audio_1, sr_1 = torchaudio.load(path_1, channels_first=False)
    audio_1 = model.audio_normalizer(audio_1, sr_1).unsqueeze(0)

    audio_2, sr_2 = torchaudio.load(path_2, channels_first=False)
    audio_2 = model.audio_normalizer(audio_2, sr_2).unsqueeze(0)

    score, decision = model.verify_batch(audio_1, audio_2)

    return score[0].item(), decision[0].item()



if __name__ == "__main__":
    main()