import gensim
import spacy
from scipy.stats import mannwhitneyu

from utils.dict_tagger import DictionaryTagger
from utils.pre_processing import read_input_file, group_to_corpuses
from utils.splitter import POSTagger, Splitter

NRC_EMOTION_LEXICON_PATH = "Input/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
nlp = spacy.load('en', disable=['parser', 'ner'])


def get_sentiment(sentiment):
    if not isinstance(sentiment, tuple):
        return dict()
    results = dict()
    results.update({sentiment[0]: sentiment[1]})
    return results


def sentiment_score(dict_tagged_sentences):
    emotions = dict()
    for sentence in dict_tagged_sentences:
        for token in sentence:
            for tag in token[2]:
                value = get_sentiment(tag)
                if not value:
                    continue
                emotions.update({tag[0]: emotions.get(tag[0], 0) + value.get(tag[0], 0)})
    return emotions


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join(
            [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


def mann_whitney_u_test(group_pa, group_yt):
    print("PA Sentiment: %s" % group_pa)
    print("YT Sentiment: %s" % group_yt)
    print("Mann Whitney-u Test:")
    pa_count = list(group_pa.values())
    yt_count = list(group_yt.values())
    try:
        mw_stat, mw_p = mannwhitneyu(pa_count, yt_count)
    except ValueError:
        mw_stat = -1  # in case of ties, Mann-Whitney cannot rank, and so cannot calculate U
        mw_p = -1

    print("MannWhitney U Value: %s" % mw_stat)
    print("MannWhitney rho Value: %s" % mw_p)


if __name__ == "__main__":
    filename = 'Input/US3_ALL_TRANSCRIPTS.docx'
    lines = read_input_file(filename)
    pa_group, yt_group = group_to_corpuses(lines)

    pa_group = lemmatization(sent_to_words(pa_group))
    pa_group = ". ".join(pa_group)
    splitter = Splitter()
    postagger = POSTagger()

    splitted_sentences = splitter.split(pa_group)

    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

    dicttagger = DictionaryTagger(NRC_EMOTION_LEXICON_PATH)

    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    pa_sentiment = sentiment_score(dict_tagged_sentences)

    yt_group = lemmatization(sent_to_words(yt_group))

    yt_group = ". ".join(yt_group)
    splitter = Splitter()
    postagger = POSTagger()
    splitted_sentences = splitter.split(yt_group)

    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

    dicttagger = DictionaryTagger(NRC_EMOTION_LEXICON_PATH)

    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    yt_sentiment = sentiment_score(dict_tagged_sentences)
    mann_whitney_u_test(pa_sentiment, yt_sentiment)
