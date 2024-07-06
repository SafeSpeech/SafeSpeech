import torch
import bert_vits2.commons as commons
from bert_vits2.text import cleaned_text_to_sequence
from bert_vits2.text.cleaner import clean_text
import bert_vits2.utils as utils
import numpy as np

from bert_vits2.models import SynthesizerTrn
from bert_vits2.text.symbols import symbols

from bert_vits2.oldVersion.V210.models import SynthesizerTrn as V210SynthesizerTrn
from bert_vits2.oldVersion.V210.text import symbols as V210symbols
from bert_vits2.oldVersion.V200.models import SynthesizerTrn as V200SynthesizerTrn
from bert_vits2.oldVersion.V200.text import symbols as V200symbols
from bert_vits2.oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn
from bert_vits2.oldVersion.V111.text import symbols as V111symbols
from bert_vits2.oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn
from bert_vits2.oldVersion.V110.text import symbols as V110symbols
from bert_vits2.oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn
from bert_vits2.oldVersion.V101.text import symbols as V101symbols

from bert_vits2.oldVersion import V111, V110, V101, V200, V210

latest_version = "2.2"

SynthesizerTrnMap = {
    "2.1": V210SynthesizerTrn,
    "2.0.2-fix": V200SynthesizerTrn,
    "2.0.1": V200SynthesizerTrn,
    "2.0": V200SynthesizerTrn,
    "1.1.1-fix": V111SynthesizerTrn,
    "1.1.1": V111SynthesizerTrn,
    "1.1": V110SynthesizerTrn,
    "1.1.0": V110SynthesizerTrn,
    "1.0.1": V101SynthesizerTrn,
    "1.0": V101SynthesizerTrn,
    "1.0.0": V101SynthesizerTrn,
}

symbolsMap = {
    "2.1": V210symbols,
    "2.0.2-fix": V200symbols,
    "2.0.1": V200symbols,
    "2.0": V200symbols,
    "1.1.1-fix": V111symbols,
    "1.1.1": V111symbols,
    "1.1": V110symbols,
    "1.1.0": V110symbols,
    "1.0.1": V101symbols,
    "1.0": V101symbols,
    "1.0.0": V101symbols,
}


# def get_emo_(reference_audio, emotion, sid):
#     emo = (
#         torch.from_numpy(get_emo(reference_audio))
#         if reference_audio and emotion == -1
#         else torch.FloatTensor(
#             np.load(f"emo_clustering/{sid}/cluster_center_{emotion}.npy")
#         )
#     )
#     return emo


def get_net_g(model_path: str, version: str, device: str, hps):
    if version != latest_version:
        net_g = SynthesizerTrnMap[version](
            len(symbolsMap[version]),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    else:
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g


def get_text(text, language_str, bert, hps, device):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # bert_ori = get_bert(norm_text, word2ph, language_str, device)
    bert_ori = bert[language_str].get_bert_feature(norm_text, word2ph, device)
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


def infer(
    text,
    emotion,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    bert=None,
    clap=None,
    reference_audio=None,
    skip_start=False,
    skip_end=False,
):
    inferMap_V3 = {
        "2.1": V210.infer,
    }
    inferMap_V2 = {
        "2.0.2-fix": V200.infer,
        "2.0.1": V200.infer,
        "2.0": V200.infer,
        "1.1.1-fix": V111.infer_fix,
        "1.1.1": V111.infer,
        "1.1": V110.infer,
        "1.1.0": V110.infer,
    }
    inferMap_V1 = {
        "1.0.1": V101.infer,
        "1.0": V101.infer,
        "1.0.0": V101.infer,
    }
    version = hps.version if hasattr(hps, "version") else latest_version
    if version != latest_version:
        if version in inferMap_V3.keys():
            return inferMap_V3[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                language,
                hps,
                net_g,
                device,
                reference_audio,
                emotion,
                skip_start,
                skip_end,
            )
        if version in inferMap_V2.keys():
            return inferMap_V2[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                language,
                hps,
                net_g,
                device,
            )
        if version in inferMap_V1.keys():
            return inferMap_V1[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                hps,
                net_g,
                device,
            )
    if isinstance(reference_audio, np.ndarray):
        emo = clap.get_clap_audio_feature(reference_audio, device)
    else:
        emo = clap.get_clap_text_feature(emotion, device)
    emo = torch.squeeze(emo, dim=1)

    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text, language, bert, hps, device
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        emo = emo.to(device).unsqueeze(0)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                emo,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio


def infer_multilang(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    bert=None,
    clap=None,
    reference_audio=None,
    emotion=None,
    skip_start=False,
    skip_end=False,
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
    # emo = get_emo_(reference_audio, emotion, sid)
    if isinstance(reference_audio, np.ndarray):
        emo = clap.get_clap_audio_feature(reference_audio, device)
    else:
        emo = clap.get_clap_text_feature(emotion, device)
    emo = torch.squeeze(emo, dim=1)
    for idx, (txt, lang) in enumerate(zip(text, language)):
        skip_start = (idx != 0) or (skip_start and idx == 0)
        skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)
        (
            temp_bert,
            temp_ja_bert,
            temp_en_bert,
            temp_phones,
            temp_tones,
            temp_lang_ids,
        ) = get_text(txt, lang, bert, hps, device)
        if skip_start:
            temp_bert = temp_bert[:, 3:]
            temp_ja_bert = temp_ja_bert[:, 3:]
            temp_en_bert = temp_en_bert[:, 3:]
            temp_phones = temp_phones[3:]
            temp_tones = temp_tones[3:]
            temp_lang_ids = temp_lang_ids[3:]
        if skip_end:
            temp_bert = temp_bert[:, :-2]
            temp_ja_bert = temp_ja_bert[:, :-2]
            temp_en_bert = temp_en_bert[:, :-2]
            temp_phones = temp_phones[:-2]
            temp_tones = temp_tones[:-2]
            temp_lang_ids = temp_lang_ids[:-2]
        bert.append(temp_bert)
        ja_bert.append(temp_ja_bert)
        en_bert.append(temp_en_bert)
        phones.append(temp_phones)
        tones.append(temp_tones)
        lang_ids.append(temp_lang_ids)
    bert = torch.concatenate(bert, dim=1)
    ja_bert = torch.concatenate(ja_bert, dim=1)
    en_bert = torch.concatenate(en_bert, dim=1)
    phones = torch.concatenate(phones, dim=0)
    tones = torch.concatenate(tones, dim=0)
    lang_ids = torch.concatenate(lang_ids, dim=0)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        emo = emo.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                emo,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio
