from utils.pre_processing import read_input_file, group_to_corpuses, remove_special_characters_from_lines


def lexical_diversity_analyser(words_grouping):
    return len(set(words_grouping)) / len(words_grouping)


def main():
    filename = 'Input/US3_ALL_TRANSCRIPTS.docx'
    lines = read_input_file(filename)  # read input docx file
    pa_group, yt_group, _, _ = group_to_corpuses(lines)  # grouping into two corpuses and removing the unnecessary lines

    # tokenize, remove special characters, stem and lemmatize
    pa_cleaned_up, _ = remove_special_characters_from_lines(pa_group)

    yt_cleaned_up, _ = remove_special_characters_from_lines(yt_group)

    # calculate the lexical diversity
    pa_ld = lexical_diversity_analyser(pa_cleaned_up)
    yt_ld = lexical_diversity_analyser(yt_cleaned_up)

    print("PA lexical diversity: %s" % pa_ld)
    print("YT lexical diversity: %s" % yt_ld)
    print("PA>YT? %s" % (pa_ld > yt_ld))


if __name__ == "__main__":
    main()
