import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sounddataloader import get_loader

from model import Model_CNN, Model_CNN_1D


def fit(epoch, model, optimizer, criterion, device, dataloader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0
    running_correct = 0

    for data, target in tqdm(dataloader):
        data = data.unsqueeze(1).float().to(device)
        target = target.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            output = model(data)
        else:
            with torch.no_grad():
                output = model(data)

        # compute loss
        loss = criterion(output, target)
        running_loss += loss.item()
        predicts = torch.argmax(output, dim=1)
        running_correct += (predicts == target.long()).sum()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_correct.item() / len(dataloader.dataset)
    print('[%d][%s] loss: %.4f acc: %.4f' % (epoch, phase, epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc


def train(path_dataset, batch_size, device, model, epochs, lr):
    print('loading data ...........')
    train_loader, val_loader = get_loader(path_dataset, batch_size=batch_size)

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 1)

    criterion = torch.nn.CrossEntropyLoss()

    print('start training ..........')
    train_losses, train_acc = [], []
    val_losses, val_acc = [], []
    for epoch in range(epochs):
        scheduler.step(epoch)

        train_epoch_loss, train_epoch_acc \
            = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss, val_epoch_acc \
            = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_acc > np.max(val_acc):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_losses.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        write_figures('output/', train_losses, val_losses, train_acc, val_acc)


def write_figures(location, train_losses, val_losses, train_accuracy, val_accuracy):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')
    plt.plot(train_accuracy, label='training accuracy')
    plt.plot(val_accuracy, label='validation accuracy')
    plt.legend()
    plt.savefig(location + '/accuracy.png')
    plt.close('all')


if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_cnn = Model_CNN_1D().to(device)
    num_epochs = 2000
    learning_rate = 0.1
    batch_size = 32
    path_dataset = 'data/radvess/trainval'
    train(path_dataset, batch_size, device, model_cnn, num_epochs, learning_rate)