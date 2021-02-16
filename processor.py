import codecs

import numpy as np
import torch
from torch.utils.data import random_split


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
    """
    Creating a corpus of all the tokens used
    """
    tokenized_corpus = []  # Let us put the tokenized corpus in a list
    for sentence in data:
        sentence = sentence.lower()
        tokenized_sentence = []
        for token in sentence.split(' '):  # simplest split is
            tokenized_sentence.append(token)
        tokenized_corpus.append(tokenized_sentence)
    # Create single list of all vocabulary
    vocabulary = []  # Let us put all the tokens (mostly words) appearing in the vocabulary in a list
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                if True:
                    vocabulary.append(token)
    return vocabulary, tokenized_corpus


def lowercase_dataset(data):
    for sentence in data:
        sentence.lower()


def build_glove_dictionary(embedding_dim):
    word2embedding = {}
    with codecs.open('model-downloads/glove.6B.{}d.txt'.format(embedding_dim), 'r', 'utf-8') as f:
        for line in f.readlines():
            if len(line.strip().split()) > 3:
                word = line.strip().split()[0]
                word2embedding[word] = np.array(list(map(float, line.strip().split()[1:])))

    return word2embedding


def lookup_glove(word2embedding, word, embedding_dim):
    try:
        return word2embedding[word]
    except KeyError:
        return np.random.normal(size=(embedding_dim, ))


def build_embedding_tensor(vocab, embedding_dim=50):
    glove_vectors = np.zeros((len(vocab), embedding_dim))
    word2embedding = build_glove_dictionary(embedding_dim)

    for i, word in enumerate(vocab):
        glove_vectors[i] = lookup_glove(word2embedding, word, embedding_dim)

    return torch.from_numpy(glove_vectors)

# def build_glove_embedding_matrix(word2idx, max_len):
#     embeddings_index = {}
#     with open('glove.6B.100d.txt') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = coefs
#
#     embedding_matrix = np.zeros((len(word2idx) + 1, 26))
#     for word, i in word2idx.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#     return embedding_matrix


def vectorize_sentences(tokenized_corpus, vocab):
    word2idx = {w: idx + 1 for (idx, w) in enumerate(vocab)}
    # we reserve the 0 index for the padding token
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
