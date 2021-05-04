from pytorch_lightning.metrics.functional import f1
import torch


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    # based on https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


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
