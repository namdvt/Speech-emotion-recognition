import torch
from tqdm import tqdm
from sounddataloader import convert_to_mfcc, SoundDataLoader
from model import Model_CNN
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model_CNN().to(device)
    model.load_state_dict(torch.load('output/weight.pth', map_location=device))

    batch_size = 16
    root = 'data/radvess/test'

    dataset = SoundDataLoader(root=root)

    dataset = convert_to_mfcc(dataset)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)

    ytrue = torch.tensor([])
    ypred = torch.tensor([])
    running_correct = 0
    for data, target in tqdm(test_loader):
        data = data.unsqueeze(1).float().to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)

        predicts = torch.argmax(output, dim=1)
        running_correct += (predicts == target.long()).sum()
        ypred = torch.cat((ypred, predicts.cpu().float()), 0)
        ytrue = torch.cat((ytrue, target.cpu().float()), 0)

    acc = running_correct.item() / len(test_loader.dataset)
    print(metrics.confusion_matrix(ytrue, ypred.detach()))
    print(acc)
