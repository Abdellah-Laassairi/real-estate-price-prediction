import re
from string import punctuation

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk packages


# functions to determine the type of a word
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


# lemmatizer + tokenizer (+ stemming) class
class LemmaTokenizer(object):
    def __init__(self):
        # defining stopwords: using the one that comes with nltk + appending it with words seen from the above evaluation
        self.stop_words = stopwords.words('english')
        self.stop_append = ['[', "'", ']', '.', ',', '`', '"', "'", '!', ';']
        self.stop_words.extend(self.stop_append)

        # list of word types (nouns and adjectives) to leave in the text
        self.defTags = [
            'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR'
        ]  #, 'RB', 'RBS', 'RBR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

        self.wnl = WordNetLemmatizer()
        # we define (but not use) a stemming method, uncomment the last line in __call__ to get stemming tooo
        self.stemmer = nltk.stem.SnowballStemmer('english')

    def __call__(self, doc):
        # pattern for numbers | words of length=2 | punctuations | words of length=1
        pattern = re.compile(
            r'[0-9]+|\b[\w]{2,2}\b|[%.,_`!"&?\')({~@;:#}+-]+|\b[\w]{1,1}\b')

        # tokenize document
        doc_tok = word_tokenize(doc)

        #filter out patterns from words
        doc_tok = [x for x in doc_tok if x not in self.stop_words]
        doc_tok = [pattern.sub('', x) for x in doc_tok]

        # get rid of anything with length=1
        doc_tok = [x for x in doc_tok if len(x) > 1]

        # # position tagging
        doc_tagged = nltk.pos_tag(doc_tok)

        # # selecting nouns and adjectives
        doc_tagged = [(t[0], t[1]) for t in doc_tagged if t[1] in self.defTags]

        # # preparing lemmatization
        # doc = [(t[0], penn_to_wn(t[1])) for t in doc_tagged]

        # print(doc_tagged)
        # lemmatization
        doc = [self.wnl.lemmatize(t[0]) for t in doc_tagged]
        # print(doc)
        # uncomment if you want stemming as well
        doc = [self.stemmer.stem(x) for x in doc]
        return doc
