from typing import Optional
from pytorch_lightning.metrics.functional import f1
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self, alpha: Optional[Tensor] = None, gamma: float = 0., reduction: str = "mean", ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("Reduction must be one of: 'mean', 'sum', 'none'.")

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none", ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.0
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def to_device(data, device):
    if isinstance(data, tuple) or isinstance(data, list):
        result = []
        for d in data:
            result.append(d.to(device))
        return result
    return data.to(device)


def train(model, device, dataloader, epoch, criterion, optimizer):
    model.train()
    batch_count = len(dataloader)
    data_count = len(dataloader.dataset)

    loss_sum = 0
    correct_predictions = 0
    f1_sum = 0

    for batch_id, data in enumerate(dataloader):
        input_data, labels = data
        input_data, labels = to_device(input_data, device), to_device(labels, device)

        output = model(input_data)
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
            input_data, labels = data
            input_data, labels = to_device(input_data, device), to_device(labels, device)

            output = model(input_data)
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
