import argparse
import datetime
import torch
import wandb
from tqdm import tqdm
import os

from torch.utils.data import DataLoader
from torchvision import datasets
import script_utils
from covidloader import CryoETDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='treaptofun',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

        # Replace CIFAR10 with your custom 3D dataset
        train_dataset = CryoETDataset(args.root_dir, flag='Train', train_num = args.train_num, transform = script_utils.get_transform())
        train_loader = script_utils.cycle(DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2))

        test_dataset = CryoETDataset(args.root_dir, flag= 'Test',train_num = args.train_num, transform = script_utils.get_transform())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)

        max_value = train_dataset.max_value
        min_value = train_dataset.min_value
        
        acc_train_loss = 0

        for iteration in tqdm(range(1, args.iterations + 1)):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y).mean()
            else:
                loss = diffusion(x).mean()

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()

            # if iteration % args.log_rate == 0:
            #     test_loss = 0
            #     with torch.no_grad():
            #         diffusion.eval()
            #         for x, y in test_loader:
            #             x = x.to(device)
            #             y = y.to(device)

            #             if args.use_labels:
            #                 loss = diffusion(x, y)
            #             else:
            #                 loss = diffusion(x)

            #             test_loss += loss.item()

            #     if args.use_labels:
            #         samples = diffusion.sample(args.train_num, device, y=torch.arange(args.train_num, device=device))
            #     else:
            #         samples = diffusion.sample(args.train_num, device)


            #     sample = (sample + 1) * (max_value - min_value) / 2 + min_value
            #     sample = sample.detach().numpy()
            #     test_loss /= len(test_loader)
            #     acc_train_loss /= args.log_rate

                # wandb.log({
                #     "test_loss": test_loss,
                #     "train_loss": acc_train_loss,
                #     "samples": [wandb.Image(sample) for sample in samples],
                # })

                # acc_train_loss = 0
            
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
        
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=6,  # Adjust this if needed
        iterations=400000,  # Adjust this if needed

        log_to_wandb=False,
        log_rate=10000,
        checkpoint_rate=10000,
        log_dir="logs",
        project_name="ddpm",
        run_name=run_name,
        root_dir = "/home/flask-diffusion/cryoet/DDPM/cryoet_dataset/SNR001",
        train_num = 10,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()