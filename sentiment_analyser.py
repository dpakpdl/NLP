import gensim
import pandas as pd
import spacy

from personal_pronoun_analyser import mann_whitney_u_test
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


if __name__ == "__main__":
    filename = 'Input/US3_ALL_TRANSCRIPTS.docx'
    lines = read_input_file(filename)
    pa_grouping, yt_grouping, pa_grouped_by_participant, yt_grouped_by_participant = group_to_corpuses(lines)

    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger(NRC_EMOTION_LEXICON_PATH)

    pa_sentiment_full = dict()
    yt_sentiment_full = dict()

    # iterate throught sentences in each participant to find the score for emotions/sentiments
    for participant, pa_sentences in pa_grouped_by_participant.items():
        # lemmatization of tokenized sentences in each participants in pa
        pa_sentences_group = lemmatization(sent_to_words(pa_sentences))
        pa_sentences_group = ". ".join(pa_sentences_group)

        # split the paragraphs pa corpus into sentences and each sentence are tokenized to words list
        splitted_sentences = splitter.split(pa_sentences_group)

        # POS tagging of tokenized words in sentences
        pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

        # associating emotions and sentiment to the words in PA corpus using the loaded word emotion-sentiment
        # from NRC emmotion lexicon
        dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

        # calculate the scores for each participant in pa
        pa_sentiment = sentiment_score(dict_tagged_sentences)

        # get sentences for that participant from yt corpus
        yt_sentences = yt_grouped_by_participant.get(participant, [])

        # lemmatization of tokenized sentences for that participants in pa
        yt_sentences_group = lemmatization(sent_to_words(yt_sentences))

        yt_sentences_group = ". ".join(yt_sentences_group)

        # split the paragraphs yt corpus into sentences and each sentence are tokenized to words list
        splitted_sentences = splitter.split(yt_sentences_group)

        # POS tagging of tokenized words in sentences
        pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

        # associating emotions and sentiment to the words in YT corpus using the loaded word emotion-sentiment
        # from NRC emmotion lexicon
        dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

        # calculate the scores for that participant in yt
        yt_sentiment = sentiment_score(dict_tagged_sentences)

        # iterate through each emotion/sentiment to get the final value
        for emotion, score in yt_sentiment.items():
            # get scores vector for each emotion/sentiment from each participant in yt
            yt_scores = yt_sentiment_full.get(emotion, [])
            yt_scores.append(score)
            yt_sentiment_full.update({emotion: yt_scores})

            # get scores vector for each emotion/sentiment from each participant in pa
            pa_scores = pa_sentiment_full.get(emotion, [])
            pa_scores.append(pa_sentiment.get(emotion, 0))
            pa_sentiment_full.update({emotion: pa_scores})
    output = mann_whitney_u_test(pa_sentiment_full, yt_sentiment_full)
    pd.DataFrame.from_dict({i: output[i] for i in output.keys()}, orient='index')
