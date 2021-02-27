import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.common_utils import set_gpu, train, eval, model_performance
from utils.dataloaders import *
from utils.processor import *


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_channels, window_size, out_dim, dropout):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv = nn.Conv2d(
            in_channels=1, out_channels=output_channels,
            kernel_size=(window_size, embedding_dim))

        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(output_channels, 16)
        self.fc2 = nn.Linear(16, out_dim)

    def forward(self, x):
        x_embed = self.embedding(x)
        x_embed = x_embed.unsqueeze(1)
        x_embed = self.conv(x_embed)

        x_embed = x_embed.squeeze(3)

        x_embed = F.relu(x_embed)
        x_embed = F.max_pool1d(x_embed, x_embed.shape[2])

        x_embed = x_embed.squeeze(2)

        x_embed = self.drop(x_embed)
        x_embed = self.fc1(x_embed)
        x_embed = self.drop(x_embed)
        out = self.fc2(x_embed)

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

    EMBEDDING_DIM = 100
    model = CNN(len(joint_vocab), EMBEDDING_DIM, 3, 5, 1, 0.2)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    train(train_loader, validation_loader, model, 20, optimizer, loss_fn, device)

    # Test data
    final_testing_df = pd.read_csv('data/task-1/truth_task_1.csv')
    final_testing_data = final_testing_df['original']
    final_testing_edits = final_testing_df['edit']
    final_testing_grades = final_testing_df['meanGrade']
    final_edited_testing = pd.Series(create_edited_sentences(final_testing_data, final_testing_edits))

    test_vocab, test_tokenized_corpus = create_vocab(final_edited_testing)

    test_vector_sentences = vectorize_sentences(test_tokenized_corpus, test_vocab)
    test_dataset = Task1Dataset(test_vector_sentences, final_testing_grades)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                              collate_fn=collate_fn_padd)

    _, _, preds, targets = eval(test_loader, model, device, loss_fn)

    test_mse, test_rmse = model_performance(preds, targets)
    print(f'| Test Set MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} |')

