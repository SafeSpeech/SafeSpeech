import logging

import regex as re

from bert_vits2.tools.classify_language import classify_language, split_alpha_nonalpha


def check_is_none(item) -> bool:
    """none -> True, not none -> False"""
    return (
        item is None
        or (isinstance(item, str) and str(item).isspace())
        or str(item) == ""
    )


def markup_language(text: str, target_languages: list = None) -> str:
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    sentences = re.split(pattern, text)

    pre_lang = ""
    p = 0

    if target_languages is not None:
        sorted_target_languages = sorted(target_languages)
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences

    for sentence in sentences:
        if check_is_none(sentence):
            continue

        lang = classify_language(sentence, target_languages)

        if pre_lang == "":
            text = text[:p] + text[p:].replace(
                sentence, f"[{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{lang.upper()}]")
        elif pre_lang != lang:
            text = text[:p] + text[p:].replace(
                sentence, f"[{pre_lang.upper()}][{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{pre_lang.upper()}][{lang.upper()}]")
        pre_lang = lang
        p += text[p:].index(sentence) + len(sentence)
    text += f"[{pre_lang.upper()}]"

    return text


def split_by_language(text: str, target_languages: list = None) -> list:
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    sentences = re.split(pattern, text)

    pre_lang = ""
    start = 0
    end = 0
    sentences_list = []

    if target_languages is not None:
        sorted_target_languages = sorted(target_languages)
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences

    for sentence in sentences:
        if check_is_none(sentence):
            continue

        lang = classify_language(sentence, target_languages)

        end += text[end:].index(sentence)
        if pre_lang != "" and pre_lang != lang:
            sentences_list.append((text[start:end], pre_lang))
            start = end
        end += len(sentence)
        pre_lang = lang
    sentences_list.append((text[start:], pre_lang))

    return sentences_list


def sentence_split(text: str, max: int) -> list:
    pattern = r"[!(),—+\-.:;?？。，、；：]+"
    sentences = re.split(pattern, text)
    discarded_chars = re.findall(pattern, text)

    sentences_list, count, p = [], 0, 0

    # 按被分割的符号遍历
    for i, discarded_chars in enumerate(discarded_chars):
        count += len(sentences[i]) + len(discarded_chars)
        if count >= max:
            sentences_list.append(text[p : p + count].strip())
            p += count
            count = 0

    if p < len(text):
        sentences_list.append(text[p:])

    return sentences_list


def sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None):
    if speaker_lang is not None and len(speaker_lang) == 1:
        if lang.upper() not in ["AUTO", "MIX"] and lang.lower() != speaker_lang[0]:
            logging.debug(
                f'lang "{lang}" is not in speaker_lang {speaker_lang},automatically set lang={speaker_lang[0]}'
            )
        lang = speaker_lang[0]

    sentences_list = []
    if lang.upper() != "MIX":
        if max <= 0:
            sentences_list.append(
                markup_language(text, speaker_lang)
                if lang.upper() == "AUTO"
                else f"[{lang.upper()}]{text}[{lang.upper()}]"
            )
        else:
            for i in sentence_split(text, max):
                if check_is_none(i):
                    continue
                sentences_list.append(
                    markup_language(i, speaker_lang)
                    if lang.upper() == "AUTO"
                    else f"[{lang.upper()}]{i}[{lang.upper()}]"
                )
    else:
        sentences_list.append(text)

    for i in sentences_list:
        logging.debug(i)

    return sentences_list