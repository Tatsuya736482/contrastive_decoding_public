import collections
import itertools
import re
import argparse
import os, copy
import json

re_sentence = re.compile(r"[^。．！？!?]+[。．！？!?]?")
re_kanji = re.compile(
    r"[々〇〻\u3400-\u9FFF\uF900-\uFAFF]|[\uD840-\uD87F][\uDC00-\uDFFF]"
)
re_space = re.compile(r"\s+")


def text_to_doc(text):
    D = []
    for line in text.split("\n"):
        if line == "":
            continue
        D.append(re_sentence.findall(line))
    return D


def count_duplicates(X):
    dup_items = 0
    dup_letters = 0

    C = collections.Counter(X)
    for s, f in C.items():
        dup_items += f - 1
        dup_letters += len(s) * (f - 1)

    return dup_items, dup_letters


def ngram(s, n):
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def count_japanese_letters(s):
    num_hiragana = 0
    num_katakana = 0
    num_toten = 0
    num_kuten = 0
    for c in s:
        o = ord(c)
        if 0x3041 <= o <= 0x3096:
            num_hiragana += 1
        elif 0x30A1 <= o <= 0x30FA:
            num_katakana += 1
        elif c in ("、", "，"):
            num_toten += 1
        elif c in ("。", "．", "！", "？"):
            num_kuten += 1
    num_kanji = len(re_kanji.findall(s))
    return num_hiragana, num_katakana, num_kanji, num_kuten, num_toten


def duplicate_fraction(text):

    r = {}

    for n in range(2, 5):
        C = collections.Counter(ngram(text, n))
        if not C:
            top_ngram, top_freq = "", 0
        else:
            top_ngram, top_freq = C.most_common(1)[0]
        total_freq = sum(f for _, f in C.items())
        r[f"top_{n}gram_character_fraction"] = (
            top_freq / total_freq if 0 < total_freq else 0.0
        )

    # top_100gramを追加
    C = collections.Counter(ngram(text, 100))
    if not C:
        top_ngram, top_freq = "", 0
    else:
        top_ngram, top_freq = C.most_common(1)[0]
    total_freq = sum(f for _, f in C.items())
    r[f"top_100gram_character_frequency"] = top_freq

    return r

def apply_v2(Q):
    violations = []

    if Q["top_2gram_character_fraction"] >= 0.16:
        violations.append("top_2gram_character_fraction")
    if Q["top_3gram_character_fraction"] >= 0.18:
        violations.append("top_3gram_character_fraction")
    if Q["top_4gram_character_fraction"] >= 0.20:
        violations.append("top_4gram_character_fraction")
    if Q["top_100gram_character_frequency"] >= 10:
        violations.append("top_100gram_character_frequency")

    result = len(violations) != 0
    return result


def check_repetition(text):
    if len(text.strip()) < 100:
        return False
    Q = duplicate_fraction(text)
    return apply_v2(Q)
