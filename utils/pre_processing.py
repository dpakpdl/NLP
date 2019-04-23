#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

import docx
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

from files import *

sys.getdefaultencoding()

verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
nouns = ['NNS', 'NNPS']  # + 'NN' + 'NNP',
adjectives = ['JJ', 'JJR', 'JJS']

OUTPUT_PATH = "Output"
PA_WORD_FREQUENCY_CSV_FILENAME = "PaCorpusFrequencies.csv"
YT_WORD_FREQUENCY_CSV_FILENAME = "YtCorpusFrequencies.csv"
CORPUS_COMPARISON_FILENAME = "CorpusCompare.csv"


def read_input_file(file_path):
    doc = docx.Document(file_path)
    total_lines = list()
    for i in doc.paragraphs:
        total_lines.append(i.text)
    return total_lines


def group_to_corpuses(lines_in_input):
    yt_group = list()
    pa_group = list()
    initial_group_flag = None
    regex = re.compile('^P[0-9]+$')

    for line in lines_in_input:
        if not line.strip():
            continue
        if line.strip().startswith("Joni: ") or line.strip().startswith("Jim:"):
            continue
        if line.strip().startswith("R:") or line.strip().startswith("R: ") or line.strip().startswith("R :"):
            continue
        if "R:" in line:
            lines_list = line.splitlines()
            for single_line in lines_list:
                if single_line.startswith("R:"):
                    lines_list.remove(single_line)
            line = ",".join(lines_list)

        if re.match(regex, line):
            continue
        if line.strip().startswith('2018-11-') or line.strip().startswith('Total experiment talk time:'):
            continue
        line = re.sub(r'\[[^()]*\]', '', line)  # regex removing text between brackets

        line = line.replace('...', ' ,')
        line = line.replace('…', ' ,')

        if not line.strip().startswith("P:") and not line.strip().startswith('YT') and not line.strip().startswith(
                'PA'):
            line = "".join([i if ord(i) < 128 else ' ' for i in line])
            if line.strip().startswith("R :") or line.strip().startswith("R4.") or line.strip().startswith("P10"):
                continue

        if line.strip().lower() == 'yt':
            initial_group_flag = 'yt'
            continue
        elif line.strip().lower() == 'pa':
            initial_group_flag = 'pa'
            continue
        elif line:
            if initial_group_flag == 'pa':
                pa_group.append(line)
            else:
                yt_group.append(line)
    return pa_group, yt_group


def remove_special_characters_from_lines(lines):
    total_clean_word_list = list()
    line_wise_word_list = list()
    for sentence in lines:
        tokens = nltk.word_tokenize(sentence)
        if 'P' in tokens:
            tokens.remove('P')
        tokens = list(filter(lambda x: x, map(lambda x: re.sub(r'[^A-Za-z0-9]+', '', x), tokens)))
        # remove articles
        tokens = [token for token in tokens if token.lower() not in ['a', 'an', 'the']]
        tagged = nltk.pos_tag(tokens)

        # lemmatize and stem the words
        stemmer = nltk.stem.PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        clean_word_list = list()
        for x, y in tagged:
            x = x.lower()
            if y in adjectives:
                clean_word_list.append(lemmatizer.lemmatize(x, pos='a'))
            elif y in verbs:
                clean_word_list.append(lemmatizer.lemmatize(x, pos='v'))
            elif y in nouns:
                clean_word_list.append(stemmer.stem(x))
            else:
                clean_word_list.append(x)
        total_clean_word_list.extend(clean_word_list)
        line_wise_word_list.append(clean_word_list)
    return total_clean_word_list, line_wise_word_list


def write_to_frequency_file(filename, frequencies):
    freq_file = os.path.join(OUTPUT_PATH, filename)
    docs = dict()
    docs[filename] = frequencies
    df = pd.DataFrame(docs)
    df = df.fillna(0)
    df.to_csv(freq_file, encoding="utf-8")  # write out to CSV


def write_to_corpus_file(data, _type=PA_CORPUS_TEXT):
    text_file_path = os.path.join(OUTPUT_PATH, _type)
    with open(text_file_path, 'w') as outfile:
        for line in data:
            outfile.write("%s\n" % line)


if __name__ == "__main__":
    filename = 'Input/US3_ALL_TRANSCRIPTS.docx'
    lines = read_input_file(filename)
    pa_group_, yt_group_ = group_to_corpuses(lines)
    # write_to_corpus_file(pa_group, PA_CORPUS_TEXT)
    # write_to_corpus_file(yt_group, YT_CORPUS_TEXT)
    pa_cleaned_up = remove_special_characters_from_lines(pa_group_)
    yt_cleaned_up = remove_special_characters_from_lines(yt_group_)
