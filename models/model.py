from torchvision.models import resnet18
from torchvision import transforms
from skimage import transform, io
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import torch


class Model(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.regressor = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_features=1000),
            torch.nn.Linear(1000, 1),
        )
    
    def forward(self, X):
        return self.regressor(self.resnet(X))

class CACDDataset(Dataset):
    def __init__(self, root_path):
        file_names_path = os.path.join(root_path, "CACD_file_names_split.pickle")
        ages_path = os.path.join(root_path, "CACD_ages_split.pickle")

        self.root = os.path.abspath(os.path.join(root_path, "../CACD2000/"))
        self.totensor = transforms.ToTensor()
        with open(file_names_path, "rb") as f:
            self.file_names = np.array(pickle.load(f))

        with open(ages_path, "rb") as f:
            self.ages = pickle.load(f)
            
    def __len__(self):
        return len(self.ages)
    
    def __getitem__(self, idx):
        if type(idx) == slice:
            idx = list(range(*idx.indices(len(self))))
                
        if type(idx) == list:
            idxs = idx
            images = []
            ages = []
            for idx in idxs:
                img_path = os.path.join(self.root, self.file_names[idx])
                img = self.totensor(transform.resize(io.imread(img_path), (224, 224))).float()
                age = self.ages[idx]
                images.append(img)
                ages.append(age)
            return torch.stack(images), torch.Tensor(ages).view(-1, 1)
        else:
            img_path = os.path.join(self.root, self.file_names[idx])
            img = self.totensor(transform.resize(io.imread(img_path), (224, 224)))
            age = self.ages[idx]
            return img, age

class Loss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, y):
        y_pred = self.model(X.float()).view(-1)
        return torch.nn.MSELoss()(y_pred, y.view(-1).float())


Opt = torch.optim.Adam
DatasetClass = CACDDataset




