import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer, BertModel
from utils.dataloaders import *
from utils.processor import *
from utils.common_utils import set_gpu, model_performance, eval


def tokenize(corpus, tokenizer):
    return [tokenizer.tokenize(sentence) for sentence in corpus]


def to_ids(corpus, tokenizer):
    return [tokenizer.convert_tokens_to_ids(sentence) for sentence in corpus]


class BertGradePredictor(nn.Module):
    def __init__(self, bert_model, total_layers, hid_size, out_size, isBidir, drop):
        super().__init__()

        self.bert_model = bert_model

        self.isBidir = isBidir

        embed_size = bert_model.config.to_dict()['hidden_size']

        if total_layers < 3:
            drop = 0

        hid_output_size = hid_size
        if isBidir:
            hid_output_size = hid_output_size * 2

        self.drop = drop

        self.gru = nn.GRU(input_size=embed_size,
                          hidden_size=hid_size,
                          num_layers=total_layers,
                          bidirectional=isBidir,
                          batch_first=True,
                          dropout=drop)

        self.fc1 = nn.Linear(hid_output_size, out_size)

    def forward(self, x):

        isBidir = self.isBidir

        with torch.no_grad():
            x_embed = self.bert_model(x)
            x_embed = x_embed[0]

        cell, hid = self.gru(x_embed)
        hid_last = hid[-1, :, :]
        hid_snd_last = hid[-2, :, :]

        if isBidir:
            hid = F.dropout(torch.cat((hid_snd_last, hid_last), dim=1), self.drop)
        else:
            hid = F.dropout(hid_last, self.drop)

        out = self.fc1(hid)

        return out


batch_size = 64
learning_rate = 0.001
total_layers = 3
hid_size = 128
out_size = 1
drop = 0.3
isBidir = True


def train(train_loader, validation_loader, model, number_epoch, optimizer, loss_fn, device):
    """
    Training loop for the model, which calls on eval to evaluate after each epoch
    """
    print("Training model.")
    for epoch in range(1, number_epoch + 1):
        model.train()
        epoch_loss = 0
        epoch_sse = 0
        no_observations = 0  # Observations used for training so far
        for batch in train_loader:
            feature, target = batch
            feature, target = feature.to(device), target.to(device)
            no_observations = no_observations + target.shape[0]
            predictions = model(feature).squeeze(1)
            optimizer.zero_grad()
            loss = loss_fn(predictions, target)
            sse, __ = model_performance(predictions.detach().cpu().numpy(), target.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * target.shape[0]
            epoch_sse += sse

        epoch_loss, epoch_mse = epoch_loss / no_observations, epoch_sse / no_observations
        print(
            f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train MSE: {epoch_mse:.2f} | Train RMSE: {epoch_mse ** 0.5:.2f} |')


def run_this_experiment():
    device = set_gpu()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer.vocab)
    max_sentence_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)

    train_df = pd.read_csv('data/task-1/train.csv')
    test_df = pd.read_csv('data/task-1/dev.csv')

    training_data = train_df['original']
    training_edits = train_df['edit']
    test_data = test_df['original']
    test_edits = test_df['edit']

    training_grades = train_df['meanGrade']

    edited_training = pd.Series(create_edited_sentences(training_data, training_edits))
    edited_test = pd.Series(create_edited_sentences(test_data, test_edits))

    training_tokens = tokenize(edited_training, tokenizer)
    testing_tokens = tokenize(edited_test, tokenizer)

    training_ids = to_ids(training_tokens, tokenizer)
    testing_tokens = to_ids(testing_tokens, tokenizer)

    train = Task1Dataset(training_ids, training_grades)
    train_dataset, validation_dataset = dataset_split(train)

    bert_model = BertModel.from_pretrained('bert-base-uncased')

    total_layers_list = [3, 5]
    hid_size_list = [64, 256]
    drop_list = [0.2, 0.4]
    batch_size_list = [64, 256]
    learning_rates = [0.01, 0.001]

    best_mse = 10000

    for batch_size in batch_size_list:
        for hid_size in hid_size_list:
            for total_layers in total_layers_list:
                for drop in drop_list:
                    for learning_rate in learning_rates:
                        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                                                   collate_fn=collate_fn_padd)
                        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                                        collate_fn=collate_fn_padd)

                        model = BertGradePredictor(bert_model,
                                                   total_layers,
                                                   hid_size,
                                                   out_size,
                                                   isBidir,
                                                   drop)

                        model = model.to(device)

                        bert_layers = model.named_parameters()
                        bert_layers = [(layer, parameter) for layer, parameter in bert_layers]
                        for i in range(len(bert_layers)):
                            layer_p = bert_layers[i]
                            layer = layer_p[0]
                            p = layer_p[1]
                            if "bert_model" in layer:
                                p.requires_grad = False

                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                        train(train_loader, validation_loader, model, 2, optimizer, loss_fn, device)

                        _, _, preds, labels = eval(validation_loader, model)

                        mse, rmse = model_performance(preds, labels, print_output=True)

                        print("Current Hyperparameters:")
                        print(
                            "Batch Size: {}, Hidden Size: {}, Total Layers: {}, Dropout: {}, Learning Rate: {}".format(
                                batch_size, hid_size, total_layers, drop, learning_rate))
                        print("MSE: {}, RMSE: {}".format(mse, rmse))

                        if mse < best_mse:
                            best_mse = mse
                            print("Found better hyperparameters...")
                            model_hyperparams_name = 'bert-mse{}-bs{}-hs{}-lay{}-drp{}-lr{}.pt'.format(mse, batch_size,
                                                                                                       hid_size,
                                                                                                       total_layers,
                                                                                                       drop,
                                                                                                       learning_rate)
                            torch.save(model.state_dict(), "./" + model_hyperparams_name)

                        print()
