import os
import torch
import numpy as np
from mrcfile import open as mrc_open
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

import warnings
warnings.filterwarnings("ignore")

class CryoETDataset(Dataset):
    def __init__(self, root_dir, flag, train_num, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.flag = flag
        self.train_num = train_num 

        self.samples = []
        self.labels = []
        self.label_mapping = {}

        for idx, subdir in enumerate(os.listdir(self.root_dir)):
            
            # print(subdir)
            
            if idx == self.train_num:
                break
            if subdir == ".DS_Store":
                continue

            subdir_path = os.path.join(self.root_dir, subdir, "subtomogram_mrc")
            if self.flag == "Train":
                for i in range(450):
                    file_path = os.path.join(subdir_path, f"tomotarget{i}.mrc")
                    self.samples.append(file_path)
                    self.labels.append(idx)
            
            if self.flag == "Test":
                for i in range(450,500):
                    file_path = os.path.join(subdir_path, f"tomotarget{i}.mrc")
                    self.samples.append(file_path)
                    self.labels.append(idx)

            self.label_mapping[idx] = subdir
        print(len(self.label_mapping))

        # Calculate dataset statistics
        self.min_value, self.max_value = self.get_data_stats()
        print("min value", self.min_value, "max value", self.max_value)

    def get_data_stats(self):
        min_value, max_value = np.inf, -np.inf
        for file_path in self.samples:
            with mrc_open(file_path, "r") as mrc:
                data = np.array(mrc.data)
                min_value = min(min_value, data.min())
                max_value = max(max_value, data.max())
        return min_value, max_value

    def normalize_data(self, data):
        return 2 * (data - self.min_value) / (self.max_value - self.min_value) - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path = self.samples[index]
        with mrc_open(file_path, "r") as mrc:
            data = np.array(mrc.data)

        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        data = self.normalize_data(data)

        return data, label

# test the dataloader
def main():
    root_dir = "/home/flask-diffusion/cryoet/DDPM/cryoet_dataset/SNR001"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CryoETDataset(root_dir, flag='Train',train_num = 10, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for epoch in range(10):  # 10 epochs as an example
        print(f"Epoch {epoch + 1}:")
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            print(f"Batch {batch_idx + 1} - Data shape: {data.shape}, Labels shape: {labels.shape}, Data range: {data.min().item()} - {data.max().item()}")
            print(labels)
            break
            
if __name__ == "__main__":
    main()
