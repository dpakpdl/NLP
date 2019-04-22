import csv
import os

import pandas as pd
from scipy.stats import mannwhitneyu

from files import *
from pre_processing import read_input_file, remove_special_characters_from_lines, group_to_corpuses, \
    write_to_frequency_file


def personal_pronoun_analysis(words_grouping):
    pronouns_to_count = WORD_LIST
    return dict((x, words_grouping.count(x)) for x in set(pronouns_to_count))


def mann_whitney_u_test(group_pa, group_yt):
    for key, value in group_pa.items():
        pa_count = [value]
        yt_count = [group_yt.get(key, 0)]
        try:
            mw_stat, mw_p = mannwhitneyu(pa_count, yt_count)
        except ValueError:
            mw_stat = -1  # in case of ties, Mann-Whitney cannot rank, and so cannot calculate U
            mw_p = -1
        print(key, pa_count, yt_count, mw_stat, mw_p)


def mann_whitney_u_test_with_file_write():
    pa_word_freq_file = os.path.join(OUTPUT_PATH, PA_WORD_FREQUENCY_CSV_FILENAME)
    yt_word_freq_file = os.path.join(OUTPUT_PATH, YT_WORD_FREQUENCY_CSV_FILENAME)

    df1 = pd.read_csv(pa_word_freq_file, index_col=0)  # read in the CSV
    df1.rename(columns={'Unnamed: 0': 'Text'}, inplace=True)  # add a label to the first column
    df1 = df1.fillna(0)  # replace NaNs with zeroes.

    df2 = pd.read_csv(yt_word_freq_file, index_col=0)  # read in the CSV
    df2.rename(columns={'Unnamed: 0': 'Text'}, inplace=True)  # add a label to the first column
    df2 = df2.fillna(0)  # replace NaNs with zeroes.

    total_docs = len(df1.columns) * len(df2.columns)

    # Make "dummy" rows of all zeroes for any words that only appear in one corpus and not the other
    missing_in_pa_corpus = []
    missing_in_yt_corpus = []
    for i in range(0, df1.shape[1]):
        missing_in_pa_corpus.append(0)

    for i in range(0, df2.shape[1]):
        missing_in_yt_corpus.append(0)

    corpus_comparison_file = os.path.join(OUTPUT_PATH, CORPUS_COMPARISON_FILENAME)

    # Iterate over the wordlist and the two corpora, and output to csv
    with open(corpus_comparison_file, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['word', 'Mann Whitney U Value', 'Mann Whitney rho-value'])
        for word in WORD_LIST:
            word = word.strip()
            if word in df1.index:
                counts_in_pa_corpus = df1.loc[word].values
            else:
                counts_in_pa_corpus = missing_in_pa_corpus
            if word in df2.index:
                counts_in_yt_corpus = df2.loc[word].values
            else:
                counts_in_yt_corpus = missing_in_yt_corpus
            try:
                mw_stat, mw_rho = mannwhitneyu(counts_in_pa_corpus, counts_in_yt_corpus)
            except ValueError:  # Was having problems with this earlier, so this is mainly for debugging reasons
                mw_stat = -1
                mw_rho = -1
            # print(word, counts_in_pa_corpus, counts_in_yt_corpus, mw_stat, mw_rho)
            writer.writerow([word, mw_stat, mw_rho])
    print("Mann Whitney-U test done.")


def main():
    filename = 'Input/US3_ALL_TRANSCRIPTS.docx'
    lines = read_input_file(filename)
    pa_group, yt_group = group_to_corpuses(lines)
    pa_cleaned_up = remove_special_characters_from_lines(pa_group)

    yt_cleaned_up = remove_special_characters_from_lines(yt_group)

    pa_personal_pronouns = personal_pronoun_analysis(pa_cleaned_up)
    print("PA personal pronoun: %s" % pa_personal_pronouns)
    yt_personal_pronouns = personal_pronoun_analysis(yt_cleaned_up)
    print("YT personal pronoun: %s" % yt_personal_pronouns)
    write_to_frequency_file(PA_WORD_FREQUENCY_CSV_FILENAME, pa_personal_pronouns)
    write_to_frequency_file(YT_WORD_FREQUENCY_CSV_FILENAME, yt_personal_pronouns)
    mann_whitney_u_test(pa_personal_pronouns, yt_personal_pronouns)
    mann_whitney_u_test_with_file_write()


main()
