import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class BaseModel(nn.Module):
    def __init__(self, num_classes = 7):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        # fully-connected layers
        self.fc1 = nn.Linear(in_features=50176, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

        # dropout layers
        self.dropout = nn.Dropout(0.33)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    correct = 0
    mean_loss = 0

    for i_batch, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        loss = F.cross_entropy(logits, y)
        mean_loss += loss.item()
        loss.backward()
        optimizer.step()

        # if i_batch % 100 == 0:
        #     print('[TRN] Train epoch: {}, batch: {}\tLoss: {:.4f}'.format(
        #         epoch, i_batch, loss.item()))

    trn_acc = correct / len(loader.dataset)
    mean_loss /= len(loader.dataset)
    # print('[TRN] Train accuracy: {:.2f}%'.format(100 * trn_acc))
    return mean_loss, trn_acc


def test_one_epoch(model, loader, device):
    model.eval()
    correct = 0
    mean_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            loss = F.cross_entropy(logits, y)
            mean_loss += loss.item()
    tst_acc = correct / len(loader.dataset)
    mean_loss /= len(loader.dataset)
    # print('[VAL] Validation accuracy: {:.2f}%'.format(100 * tst_acc))
    return mean_loss, tst_acc


def plot_results(trn_accs, tst_accs, trn_losses, tst_losses, epochs):

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(np.arange(1, epochs + 1), tst_accs, 'r', label='test accuracy')
    axs[0].plot(np.arange(1, epochs + 1), trn_accs, 'b', label='train accuracy')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('accuracy')

    axs[1].plot(np.arange(1, epochs + 1), tst_losses, 'r', label='test loss')
    axs[1].plot(np.arange(1, epochs + 1), trn_losses, 'b', label='train loss')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('loss')

    plt.tight_layout()
    plt.show()

    return fig, axs

def train_test_model(model, trn_loader, tst_loader, optimizer, scheduler, device, epochs, batch_size, model_name):

    trn_accs = []
    tst_accs = []
    trn_losses = []
    tst_losses = []

    epoch = 0
    # count = 0

    best_loss = float('inf')
    best_acc = 0

    while epoch < epochs:

        trn_loss, trn_acc = train_one_epoch(model, trn_loader, optimizer, device, epoch)
        trn_loss *= batch_size
        trn_losses.append(trn_loss)
        trn_accs.append(trn_acc * 100.)

        tst_loss, tst_acc = test_one_epoch(model, tst_loader, device)
        tst_loss *= batch_size
        tst_losses.append(tst_loss)
        tst_accs.append(tst_acc * 100.)

        epoch += 1

        if tst_loss < best_loss:
            best_loss = tst_loss
            best_acc = tst_acc
            # count = 0
            torch.save(model.state_dict(), f'{model_name}.pt')
        # else:
            # count += 1

        # if count > patience:
            # print("Early stopping")
            # break

        scheduler.step()

    fig, axs = plot_results(trn_accs, tst_accs, trn_losses, tst_losses, epoch)
    fig.savefig(f"{model_name}.png", dpi=300)

    return best_loss, best_acc
