from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage
import numpy as np
import pickle
import os
import torch
import skimage
import io

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
    def __init__(self, root_path, gcs=True):
        file_names_path = os.path.join(root_path, "data/CACD_filenames.pickle")
        ages_path = os.path.join(root_path, "data/CACD_ages.pickle")
        self.client = storage.Client()
        self.bucket = self.client.bucket("cacd2000")
        self.root = os.path.abspath(os.path.join(root_path, "data/CACD2000"))

        self.gcs = gcs
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
                img = self.getimage(idx)
                age = self.ages[idx]
                images.append(img)
                ages.append(age)
            return torch.stack(images), torch.Tensor(ages).view(-1, 1)
        else:
            img = self.getimage(idx)
            age = self.ages[idx]
            return img, age

    def getimage(self, idx):
        filename = self.file_names[idx]
        img_path = os.path.join(self.root, filename)
        source = io.Bytes(self.bucket.blob(filename).download_as_string()) if self.gcs \
            else os.path.join(self.root, filename)
        return self.totensor(
            skimage.transform.resize(
                skimage.io.imread(
                    source
                ), (224, 224)
            )
        ).float()


class Loss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, y):
        y_pred = self.model(X.float()).view(-1)
        return torch.nn.MSELoss()(y_pred, y.view(-1).float())


Opt = torch.optim.Adam
DatasetClass = CACDDataset




