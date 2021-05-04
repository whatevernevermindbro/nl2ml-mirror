from pytorch_lightning.metrics.functional import f1
import torch
import torch.nn as nn
import torch.nn.functional as F


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        n_classes = y_pred.size(1)
        y_true = F.one_hot(y_true, n_classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


def train(model, device, dataloader, epoch, criterion, optimizer):
    model.train()
    batch_count = len(dataloader)
    data_count = len(dataloader.dataset)

    loss_sum = 0
    correct_predictions = 0
    f1_sum = 0

    for batch_id, data in enumerate(dataloader):
        tokens, labels = data
        tokens, labels = tokens.to(device), labels.to(device)

        output = model(tokens)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        preds = output.argmax(dim=1)
        f1_sum += f1(preds, labels, output.size(1), average="weighted").item()
        correct_predictions += torch.sum((preds == labels).double()).item()

        if (batch_id + 1) % 10 == 0 or batch_id + 1 == batch_count:
            acc_score = torch.mean((preds == labels).double())
            print("train epoch {} [{}/{}] - accuracy {:.6f} - loss {:.6f}".format(
                epoch, batch_id + 1, batch_count, acc_score.item(), loss.item()
            ))

    mean_loss = loss_sum / batch_count
    mean_f1 = f1_sum / batch_count
    accuracy = correct_predictions / data_count
    return mean_loss, accuracy, mean_f1


def test(model, device, dataloader, epoch, criterion):
    model.eval()
    batch_count = len(dataloader)
    data_count = len(dataloader.dataset)

    loss_sum = 0
    correct_predictions = 0
    f1_sum = 0

    with torch.no_grad():
        for data in dataloader:
            tokens, labels = data
            tokens, labels = tokens.to(device), labels.to(device)

            output = model(tokens)
            loss = criterion(output, labels)

            loss_sum += loss.item()

            preds = output.argmax(dim=1)
            f1_sum += f1(preds, labels, output.size(1), average="weighted").item()
            correct_predictions += torch.sum((preds == labels).double()).item()

    mean_loss = loss_sum / batch_count
    mean_f1 = f1_sum / batch_count
    accuracy = correct_predictions / data_count
    print("test epoch {} - accuracy {:.6f} - f-score {:.6f} - loss {:.6f}".format(
        epoch, accuracy, mean_f1, mean_loss
    ))

    return mean_loss, accuracy, mean_f1
