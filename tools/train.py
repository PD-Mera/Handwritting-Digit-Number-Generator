import os, sys
from pathlib import Path
import logging
import time, datetime, math
import torch
from torchvision import transforms as T
from torch import nn
from torch.utils.data import DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from cfg.cfg import Config
from src.dataloader import LoadDataset
from src.model import Generator, Discriminator
from src.plot import draw_plot
torch.manual_seed(42)

                
def train(config: Config):
    train_data = LoadDataset(config)
    train_loader = DataLoader(train_data, 
                              batch_size=config.train_batch_size, 
                              shuffle=True,
                              num_workers=config.train_num_worker)
    
    device = config.device

    net_G = Generator(in_features=config.input_dims).to(device)
    net_D = Discriminator().to(device)

    optim_G = torch.optim.Adam(params=net_G.parameters(), lr = config.learning_rate_G)
    optim_D = torch.optim.Adam(params=net_D.parameters(), lr = config.learning_rate_D)

    criterion = nn.BCELoss()


    for epoch in range(config.EPOCH):
        for idx, image in enumerate(train_loader):
            # gauss: torch.Tensor
            # image: torch.Tensor
            image = image.to(device)
            gauss = torch.randn(image.size(0), config.input_dims).to(device)
            
            # Train D
            net_D.train()
            net_G.eval()
            net_D.zero_grad()

            real_D = net_D(image)
            real_label = torch.ones_like(real_D).to(device) * 0.9
            loss_D_real = criterion(real_D, real_label)

            fake_G = net_G(gauss)
            fake_D = net_D(fake_G)
            fake_label = torch.zeros_like(fake_D).to(device)
            loss_D_fake = criterion(fake_D, fake_label)
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optim_D.step()

            # train G
            net_D.eval()
            net_G.train()
            net_G.zero_grad()

            real_D = net_D(image)
            fake_G = net_G(gauss)

            fake_D = net_D(fake_G)
            real_label = torch.ones_like(real_D).to(device)

            loss_G = criterion(fake_D, real_label)
            loss_G.backward()
            optim_G.step()

        print(f"Epoch {epoch}: Loss D: {loss_D} | Loss G: {loss_G}")
        torch.save(net_G.state_dict(), os.path.join(config.exp_run_folder, config.model_savepath))

        # eval step

        net_G.eval()
        inputs = torch.randn([config.num_col * config.num_row, config.input_dims]).to(device)

        outputs = net_G(inputs)

        if epoch % 10 == 0:
            draw_plot(outputs, config, epoch=epoch)
        
        draw_plot(outputs, config, name = "last")



if __name__ == "__main__":
    config = Config()
    train(config)


            


