import os
import mrcfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import sys

# class CryoETDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.samples = []

#         for idx, subdir in tqdm(enumerate(os.listdir(root_dir))):
            
#             # if idx == 20:
#             #     break
            
#             if subdir == '.DS_Store':
#                 continue

#             subdir_path = os.path.join(root_dir, subdir, "subtomogram_mrc")

#             for i in tqdm(range(450)):
#                 file_path = os.path.join(subdir_path, f"tomotarget{i}.mrc")
#                 self.samples.append(file_path)



#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         file_path = self.samples[idx]
#         with mrcfile.open(file_path) as mrc:
#             data = torch.tensor(mrc.data, dtype=torch.float32)
#         return data

# import numpy as np
# from torch.utils.data import DataLoader

# def main():
#     dataset = CryoETDataset("/home/flask-diffusion/cryoet/DDPM/cryoet_dataset/SNR001")
#     dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

#     mean = 0.0
#     var = 0.0
#     total_samples = 0

#     for data in dataloader:
#         batch_size = data.size(0)
#         data = data.view(batch_size, -1)
#         mean += data.mean(dim=1).sum().item()
#         var += data.var(dim=1).sum().item()
#         total_samples += batch_size

#     mean /= total_samples
#     var /= total_samples
#     std = np.sqrt(var)

#     print(f"Mean: {mean}, Standard Deviation: {std}")
#     # mean: 0.041 std: 189.102

# def main2():
#     # read mrc file and print out the dimension
    
#     path = "/home/flask-diffusion/cryoet/DDPM/output/cryoet/0.mrc"
#     path = "/home/flask-diffusion/cryoet/DDPM/cryoet_dataset/SNR001/1cf5/subtomogram_mrc/tomotarget0.mrc"
#     with mrcfile.open(path) as mrc:
#         data = torch.tensor(mrc.data, dtype=torch.float32)
#         print(data.shape,data.max(),data.min())

def main3():

    mrc_file = "/home/flask-diffusion/cryoet/DDPM/ddpm2/output/mrc/9/1-0.mrc"
    x3d_file = "/home/flask-diffusion/cryoet/DDPM/ddpm2/static/protein_x3d/1-0.x3d"

    chimera_script = f"""
    open {mrc_file}
    volume #0 level 60
    export {x3d_file} format X3D
    close all
    """

    with open("chimera_script.cmd", "w") as f:
        f.write(chimera_script)

    os.system("/opt/UCSF/Chimera64-2023-03-29/bin/chimera --nogui chimera_script.cmd")
    os.remove("chimera_script.cmd")
    
if __name__ == "__main__":
    # main()
    # main2()
    main3()
