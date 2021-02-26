import codecs
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from utils.processor import create_edited_sentences


def hyphen_tokenizer(data):
    tokenized_corpus = []
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):
            for tok in token.split('-'):
                tokenized_sentence.append(tok)
        tokenized_corpus.append(tokenized_sentence)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus


def basic_tokenizer(data):
    tokenized_corpus = []
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):
            tokenized_sentence.append(token)
        tokenized_corpus.append(tokenized_sentence)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus


def hyphen_tokenizer_no_numbers(data):
    tokenized_corpus = []
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):
            for tok in token.split('-'):
                tok = re.sub("\d+", "", tok)
                tokenized_sentence.append(tok)
        tokenized_corpus.append(tokenized_sentence)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus


def build_glove_dictionary(embeddings_path):
    word2embedding = {}
    with codecs.open(embeddings_path, 'r', 'utf-8') as f:
        for line in f.readlines():
            if len(line.strip().split()) > 3:
                word = line.strip().split()[0]
                word2embedding[word] = np.array(list(map(float, line.strip().split()[1:])))
    return word2embedding


def check_words_not_in_glove(vocab, word2embedding):
    words_not_in_glove = 0
    for i, word in enumerate(vocab):
        words_not_in_glove += word not in word2embedding
    return words_not_in_glove


def stemming(data):
    porter = nltk.PorterStemmer()
    tokenized_corpus = []
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):
            for tok in token.split('-'):
                tok = re.sub("\d+", "", tok)
                tok = porter.stem(tok)
                tokenized_sentence.append(tok)
        tokenized_corpus.append(tokenized_sentence)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus


def lemmatize(data):
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    tokenized_corpus = []
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):
            for tok in token.split('-'):
                tok = re.sub("\d+", "", tok)
                tok = wordnet_lemmatizer.lemmatize(tok)
                tokenized_sentence.append(tok)
        tokenized_corpus.append(tokenized_sentence)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus


def remove_stopwords(data):
    tokenized_corpus = []
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):
            for tok in token.split('-'):
                if tok in stopwords.words('english'):
                    continue
                tok = re.sub("\d+", "", tok)
                tokenized_sentence.append(tok)
        tokenized_corpus.append(tokenized_sentence)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus


def run_this_experiment(embeddings_path):
    train_df = pd.read_csv('data/task-1/train.csv')
    test_df = pd.read_csv('data/task-1/dev.csv')

    training_data = train_df['original']
    training_edits = train_df['edit']
    test_data = test_df['original']
    test_edits = test_df['edit']

    edited_training = pd.Series(create_edited_sentences(training_data, training_edits))
    edited_test = pd.Series(create_edited_sentences(test_data, test_edits))

    word2embedding = build_glove_dictionary(embeddings_path)

    joint_vocab, _ = basic_tokenizer(pd.concat([edited_training, edited_test]))
    words_not_in_glove = check_words_not_in_glove(joint_vocab, word2embedding)
    print("Words not found in GloVe Embeddings after - spliting on whitespace: {}".format(words_not_in_glove))

    joint_vocab, _ = hyphen_tokenizer(pd.concat([edited_training, edited_test]))
    words_not_in_glove = check_words_not_in_glove(joint_vocab, word2embedding)
    print("Words not found in GloVe Embeddings after - splitting on hyphens: {}".format(words_not_in_glove))

    joint_vocab, _ = hyphen_tokenizer_no_numbers(pd.concat([edited_training, edited_test]))
    words_not_in_glove = check_words_not_in_glove(joint_vocab, word2embedding)
    print("Words not found in GloVe Embeddings after - removing digits: {}".format(words_not_in_glove))

    joint_vocab, _ = remove_stopwords(pd.concat([edited_training, edited_test]))
    words_not_in_glove = check_words_not_in_glove(joint_vocab, word2embedding)
    print("Words not found in GloVe Embeddings after - removing stopwords: {}".format(words_not_in_glove))

    joint_vocab, _ = lemmatize(pd.concat([edited_training, edited_test]))
    words_not_in_glove = check_words_not_in_glove(joint_vocab, word2embedding)
    print("Words not found in GloVe Embeddings after - lemmatizing: {}".format(words_not_in_glove))

    joint_vocab, _ = stemming(pd.concat([edited_training, edited_test]))
    words_not_in_glove = check_words_not_in_glove(joint_vocab, word2embedding)
    print("Words not found in GloVe Embeddings after - stemming: {}".format(words_not_in_glove))
