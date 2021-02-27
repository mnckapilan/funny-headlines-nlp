import numpy as np
import torch


def set_gpu():
    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def model_performance(output, target, print_output=False):
    """
    Returns SSE and MSE per batch (printing the MSE and the RMSE)
    """

    sq_error = (output - target) ** 2

    sse = np.sum(sq_error)
    mse = np.mean(sq_error)
    rmse = np.sqrt(mse)

    if print_output:
        print(f'| MSE: {mse:.4f} | RMSE: {rmse:.4f} |')

    return mse, rmse, sse


def eval(data_iter, model, device, loss_fn):
    """
    Evaluating model performance on the dev set
    """
    model.eval()
    epoch_loss = 0
    epoch_sse = 0
    pred_all = []
    trg_all = []
    no_observations = 0

    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch
            feature, target = feature.to(device), target.to(device)
            # for RNN:
            # model.batch_size = target.shape[0]
            no_observations = no_observations + target.shape[0]
            # model.hidden = model.init_hidden()

            predictions = model(feature).squeeze(1)
            loss = loss_fn(predictions, target)

            # We get the mse
            pred, trg = predictions.detach().cpu().numpy(), target.detach().cpu().numpy()
            mse, rmse, sse = model_performance(pred, trg)

            epoch_loss += loss.item()*target.shape[0]
            epoch_sse += sse
            pred_all.extend(pred)
            trg_all.extend(trg)

    return epoch_loss/no_observations, epoch_sse/no_observations, np.array(pred_all), np.array(trg_all)


def train(train_loader, validation_loader, model, number_epoch, optimizer, loss_fn, device):
    """
    Training loop for the model, which calls on eval to evaluate after each epoch
    """
    print("Training model.")
    for epoch in range(1, number_epoch+1):
        model.train()
        epoch_loss = 0
        epoch_sse = 0
        no_observations = 0  # Observations used for training so far
        for batch in train_loader:
            feature, target = batch
            feature, target = feature.to(device), target.to(device)
            # for RNN:
            # model.batch_size = target.shape[0]
            no_observations = no_observations + target.shape[0]
            # model.hidden = model.init_hidden()
            predictions = model(feature).squeeze(1)
            optimizer.zero_grad()
            loss = loss_fn(predictions, target)
            _, _, sse = model_performance(predictions.detach().cpu().numpy(), target.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*target.shape[0]
            epoch_sse += sse

        valid_loss, valid_mse, __, __ = eval(validation_loader, model, device, loss_fn)

        epoch_loss, epoch_mse = epoch_loss / no_observations, epoch_sse / no_observations
        print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.4f} | Train MSE: {epoch_mse:.4f} | Train RMSE: {epoch_mse**0.5:.4f} | \n \
        Val. Loss: {valid_loss:.4f} | Val. MSE: {valid_mse:.4f} |  Val. RMSE: {valid_mse**0.5:.4f} |')