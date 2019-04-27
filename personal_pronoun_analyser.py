import csv
import os

import pandas as pd
from scipy.stats import mannwhitneyu

from utils.files import *
from utils.pre_processing import read_input_file, remove_special_characters_from_lines, group_to_corpuses, \
    write_to_frequency_file


def personal_pronoun_analysis(words_grouping):
    pronouns_to_count = WORD_LIST
    return dict((x, words_grouping.count(x)) for x in set(pronouns_to_count))


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


def mann_whitney_u_test(group_pa, group_yt):
    print("Mann Whitney-u Test:")
    output = dict()
    for key, yt_value in group_yt.items():
        pa_value = group_pa.get(key)
        try:
            mw_stat, mw_p = mannwhitneyu(pa_value, yt_value, alternative="greater")
        except ValueError:
            mw_stat = -1  # in case of ties, Mann-Whitney cannot rank, and so cannot calculate U
            mw_p = -1
        output.update({
            key: (mw_stat, mw_p, sum(pa_value), sum(yt_value))
        })
    return output


def main():
    filename = 'Input/US3_ALL_TRANSCRIPTS.docx'
    lines = read_input_file(filename)
    pa_group, yt_group, pa_grouped_by_participant, yt_grouped_by_participant = group_to_corpuses(lines)
    pa_personal_pronoun_dict = {}
    yt_personal_pronoun_dict = {}

    # iterate participant wise in each corpus to get personal pronoun count
    for participant, pa_sentences in pa_grouped_by_participant.items():
        # remove special characters from pa sentences of a participant
        pa_cleaned_up, _ = remove_special_characters_from_lines(pa_sentences)

        # get personal pronoun count for that participant in pa corpus
        pa_personal_pronoun_count = personal_pronoun_analysis(pa_cleaned_up)

        # remove special characters from yt sentences of that participant
        yt_cleaned_up, _ = remove_special_characters_from_lines(yt_grouped_by_participant.get(participant, []))

        # get personal pronoun count for that participant in yt corpus

        yt_personal_pronoun_count = personal_pronoun_analysis(yt_cleaned_up)

        for pronoun, value in pa_personal_pronoun_count.items():
            # obtain count vector of each pronoun from all participants in pa corpus
            pa_pronoun_count = pa_personal_pronoun_dict.get(pronoun, [])
            pa_pronoun_count.append(value)
            pa_personal_pronoun_dict.update({pronoun: pa_pronoun_count})

            # obtain count vector of each pronoun from all participants in yt corpus
            yt_pronoun_count = yt_personal_pronoun_dict.get(pronoun, [])
            yt_pronoun_count.append(yt_personal_pronoun_count.get(pronoun, 0))
            yt_personal_pronoun_dict.update({pronoun: yt_pronoun_count})

    output = mann_whitney_u_test(pa_personal_pronoun_dict, yt_personal_pronoun_dict)
    pd.DataFrame.from_dict({i: output[i] for i in output.keys()}, orient='index')
    # pa_personal_pronouns = personal_pronoun_analysis(pa_cleaned_up)
    # print("PA personal pronoun: %s" % pa_personal_pronouns)
    # yt_personal_pronouns = personal_pronoun_analysis(yt_cleaned_up)
    # print("YT personal pronoun: %s" % yt_personal_pronouns)
    # write_to_frequency_file(PA_WORD_FREQUENCY_CSV_FILENAME, pa_personal_pronouns)
    # write_to_frequency_file(YT_WORD_FREQUENCY_CSV_FILENAME, yt_personal_pronouns)
    # mann_whitney_u_test_with_file_write()


main()
