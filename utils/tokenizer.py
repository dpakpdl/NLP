# this method is not used
import nltk
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
from string import punctuation
import json

nltk.download('wordnet')
nltk.download('stopwords')


def get_stop_words():
    stopwords_nltk_en = set(stopwords.words('english'))
    stopwords_punct = set(punctuation)
    with open("Input/stop_words_en.json", 'r') as infile:
        stopwords_json = json.load(infile)
    stopwords_json_en = set(stopwords_json['en'])
    return set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)


def remove_stop_words_from_word_list(word_list, stop_words):
    return [word for word in word_list if word not in stop_words]


def tokenize_words(document):
    # print(document)
    return list(map(str.lower, nltk.word_tokenize(document)))


def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'


def lemmatize_sent(text):
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in nltk.pos_tag(tokenize_words(text))]


def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    list_of_lemmas = [word for word in lemmatize_sent(text) if word not in get_stop_words() and not word.isdigit()]
    return list_of_lemmas
