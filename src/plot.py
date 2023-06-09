import os
from matplotlib import pyplot as plt
import torch
from torchvision import transforms as T

from cfg.cfg import Config

def postprocess_tensor(inputs: torch.Tensor, config: Config):
    inputs = inputs * torch.Tensor(config.mean).to(config.device) + torch.Tensor(config.std).to(config.device)
    return inputs

def draw_plot(inputs: torch.Tensor, config: Config, **kwarg):
    # inputs is torch tensor with BS x C x W x H
    draw_part = postprocess_tensor(inputs[:config.num_col * config.num_row], config)
    list_img_pil = [T.ToPILImage()(torch.reshape(draw_part[i], (28, 28))) for i in range(draw_part.size(0))]
    plt.figure()
    # plt.axis(False)
    for idx, img in enumerate(list_img_pil):
        plt.subplot(config.num_col, config.num_row, idx+1)
        plt.axis(False)
        plt.imshow(img, cmap="Greys")
        plt.tight_layout()

    savefig_name = config.savefig_name
    for key in kwarg.keys():
        savefig_name += f"_{kwarg[key]}"

    savefig_name += ".jpg"

    if config.phase == "train":
        plt.savefig(os.path.join(config.exp_run_folder, savefig_name))
    else:
        os.makedirs(config.output_dir, exist_ok=True)
        plt.savefig(os.path.join(config.output_dir, savefig_name))

    plt.close()