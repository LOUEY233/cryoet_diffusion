import argparse
import torch
import torchvision
import script_utils
from tqdm import tqdm
import os
import mrcfile
from covidloader import CryoETDataset


import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path))
        
        dataset = CryoETDataset(args.root_dir, flag='Train',train_num = args.train_num, transform = script_utils.get_transform())
        max_value = dataset.max_value
        min_value = dataset.min_value

        if args.use_labels:
            for label in range(args.train_num):
                print(label)
                y = torch.ones(args.train_num // args.train_num, dtype=torch.long, device=device) * label
                samples = diffusion.sample(args.train_num // args.train_num, device, y=y)

                for image_id in range(len(samples)):
                    
                    volume = (samples[image_id] + 1) * (max_value - min_value) / 2 + min_value
                    volume = volume.squeeze().cpu().numpy()

                    with mrcfile.new(f"{args.simage_pngave_dir}/{label}-{image_id}.mrc", overwrite=True) as mrc:
                        mrc.set_data(volume.astype('float32'))
        else:
            samples = diffusion.sample(args.train_num, device)

            for image_id in range(len(samples)):
                
                volume = (samples[image_id] + 1) * (max_value - min_value) / 2 + min_value
                volume = volume.squeeze().cpu().numpy()

                with mrcfile.new(f"{args.simage_pngave_dir}/{image_id}.mrc", overwrite=True) as mrc:
                    mrc.set_data(volume.astype('float32'))
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(device=device)  # Change img_size to 3D tuple
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/flask-diffusion/cryoet/DDPM/logs/ddpm-ddpm-2023-03-27-11-55-iteration-400000-model.pth")
    # "/home/flask-diffusion/cryoet/DDPM/logs/ddpm-ddpm-2023-03-26-00-17-iteration-160000-model.pth"
    # "/home/flask-diffusion/cryoet/DDPM/logs/ddpm-ddpm-2023-03-27-11-55-iteration-400000-model.pth"
    parser.add_argument("--simage_pngave_dir", type=str, default="/home/flask-diffusion/cryoet/DDPM/output/cryoet")
    parser.add_argument("--schedule_low", type=float, default=0.01)
    parser.add_argument("--schedule_high", type=float, default=1.0)
    parser.add_argument("--train_num", type=float, default=20)
    parser.add_argument("--root_dir", type = str , default = "/home/flask-diffusion/cryoet/DDPM/cryoet_dataset/SNR001")
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()