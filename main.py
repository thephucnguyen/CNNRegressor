import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import os

model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=1)

batchsize = 16
num_epochs = 50
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr = 0.007)

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.CenterCrop(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

class MyDataset(Dataset):
    def __init__(self, img_path, transform):
        self.dir = img_path
        self.transform = transform
        self.file_list = os.listdir(self.dir)

    def __getitem__(self, index):
        sample = Image.open(os.path.join(self.dir, self.file_list[index]))
        label = float(self.file_list[index][:-4].split('__')[-1])
        sample = self.transform(sample)

        return sample, np.array([label], dtype=np.float32)
    def __len__(self):
        return len(self.file_list)


train_ds = MyDataset('/content/googleAPI_test_5_6/blur/images', train_transforms)
val_ds = MyDataset('/content/googleAPI_test_5_6/blur/images', test_transforms)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=batchsize,
    shuffle=True,
)

val_loader = DataLoader(
    dataset=val_ds,
    batch_size=batchsize,
    shuffle=False,
)


loss_list = []
train_acc_list, val_acc_list = [], []
for epoch in range(num_epochs):
    avg_loss = 0
    count = 0
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        logits = model(features)
        logits = torch.sigmoid(logits)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        count += 1
    print(f'Epoch[{epoch+1}/{num_epochs}], loss: {avg_loss:.6f}')
