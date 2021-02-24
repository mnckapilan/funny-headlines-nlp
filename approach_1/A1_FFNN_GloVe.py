import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from utils.common_utils import set_gpu, train
from utils.dataloaders import *
from utils.processor import *


class FFNN(nn.Module):
    def __init__(self, glove_tensor, embedding_dim, vocab_size):
        super(FFNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).from_pretrained(glove_tensor)

        self.fc1 = nn.Linear(embedding_dim, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 5)
        self.relu3 = nn.ReLU()

        self.output = nn.Linear(5, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        sentence_lengths = x.ne(0).sum(1, keepdims=True)
        averaged = embedded.sum(1) / sentence_lengths
        out = self.fc1(averaged)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.output(out)
        return out


def run_this_experiment():
    device = set_gpu()

    train_df = pd.read_csv('data/task-1/train.csv')
    test_df = pd.read_csv('data/task-1/dev.csv')

    training_data = train_df['original']
    training_edits = train_df['edit']
    test_data = test_df['original']
    test_edits = test_df['edit']
    training_grades = train_df['meanGrade']

    edited_training = pd.Series(create_edited_sentences(training_data, training_edits))
    edited_test = pd.Series(create_edited_sentences(test_data, test_edits))

    training_vocab, training_tokenized_corpus = create_vocab(edited_training)
    joint_vocab, joint_tokenized_corpus = create_vocab(pd.concat([edited_training, edited_test]))

    training_vector_sentences = vectorize_sentences(training_tokenized_corpus, joint_vocab)
    training_dataset = Task1Dataset(training_vector_sentences, training_grades)

    train_dataset, validation_dataset = dataset_split(training_dataset)

    BATCH_SIZE = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE,
                                               collate_fn=collate_fn_padd)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                                    collate_fn=collate_fn_padd)

    glove_tensor, words_not_in_glove = build_embedding_tensor(joint_vocab, 100)

    EMBEDDING_DIM = 100
    model = FFNN(glove_tensor, EMBEDDING_DIM, len(joint_vocab))
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    train(train_loader, validation_loader, model, 200, optimizer, loss_fn, device)
