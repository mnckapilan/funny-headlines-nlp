import codecs
import re
from nltk import WordNetLemmatizer
import numpy as np
import torch


def create_edited_sentences(data, edits):
    edited_sentences = []
    for i, headline in enumerate(data):
        start_loc = headline.find('<')
        end_loc = headline.find('/>')
        edited_sentences.append(headline[:start_loc] + edits[i] + headline[end_loc + 2:])
    return edited_sentences


def remove_tags_sentences(data):
    clean_sentences = []
    for _, headline in enumerate(data):
        clean_sentences.append(headline.replace('<', '').replace('/>', ''))
    return clean_sentences


def create_vocab(data):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokenized_corpus = []
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):  # simplest split is
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


def build_glove_dictionary(embedding_dim):
    word2embedding = {}
    with codecs.open('glove.6B.{}d.txt'.format(embedding_dim), 'r', 'utf-8') as f:
        for line in f.readlines():
            if len(line.strip().split()) > 3:
                word = line.strip().split()[0]
                word2embedding[word] = np.array(list(map(float, line.strip().split()[1:])))
    return word2embedding


def lookup_glove(word2embedding, word, embedding_dim):
    try:
        return word2embedding[word], False
    except KeyError:
        return np.random.normal(size=(embedding_dim,)), True


def build_embedding_tensor(vocab, embedding_dim=50):
    glove_vectors = np.zeros((len(vocab) + 1, embedding_dim))
    word2embedding = build_glove_dictionary(embedding_dim)

    words_not_in_glove = 0
    for i, word in enumerate(vocab):
        glove_vec, in_glove = lookup_glove(word2embedding, word, embedding_dim)
        glove_vectors[i + 1] = glove_vec
        words_not_in_glove += in_glove

    return torch.from_numpy(glove_vectors).type(torch.float32), words_not_in_glove


def vectorize_sentences(tokenized_corpus, vocab):
    word2idx = {w: idx + 1 for (idx, w) in enumerate(vocab)}
    word2idx['<pad>'] = 0
    return [[word2idx[tok] for tok in sentence if tok in word2idx] for sentence in tokenized_corpus]


def create_input_tensors(tokenized_corpus, word2idx, grades):
    vectorized_sentences = [[word2idx[tok] for tok in sentence if tok in word2idx] for sentence in tokenized_corpus]
    sentence_lengths = [len(sentence) for sentence in vectorized_sentences]
    max_len = max(sentence_lengths)
    sentence_tensor = torch.zeros((len(vectorized_sentences), max_len)).long()
    for idx, (sentence, sentence_len) in enumerate(zip(vectorized_sentences, sentence_lengths)):
        sentence_tensor[idx, :sentence_len] = torch.LongTensor(sentence)
    score_tensor = torch.FloatTensor(grades)
    return sentence_tensor, score_tensor
