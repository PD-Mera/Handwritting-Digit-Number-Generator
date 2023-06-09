import os, shutil
import torch

class Config:
    def __init__(self, phase = "train"):
        super(Config, self).__init__()
        self.data_path = [
            "./data/mnist_png/training",
            "./data/mnist_png/testing",
        ]
        
        self.phase = phase
        self.input_dims = 100
        self.best_checkpoint = 'model.pth'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ### Image ###
        self.mean = [0.5]
        self.std = [0.5]

        ### Train config ###
        self.EPOCH = 300
        self.train_batch_size = 512
        self.train_num_worker = 8
        self.learning_rate_G = 0.0002
        self.learning_rate_D = 0.0002
        self.pretrained_weight = ''
        self.run_folder = "training_runs"
        self.exp_name = "exp"
        self.exp_number = self.get_exp_number()
        self.model_savepath = "model_best.pth"

        self.exp_run_folder = os.path.join(self.run_folder, self.exp_name + str(self.exp_number))
        if self.phase == 'train':
            os.makedirs(self.exp_run_folder)
            shutil.copy(os.path.abspath(__file__), self.exp_run_folder)
        
        ### Test config ###
        self.output_dir = "outputs"
        self.num_col = 6
        self.num_row = 6
        self.savefig_name = "sample"

    def get_exp_number(self):
        os.makedirs(self.run_folder, exist_ok=True)
        exp_number = 0
        for folder in os.listdir(self.run_folder):
            try:
                if int(folder.replace(self.exp_name, "")) >= exp_number:
                    exp_number = int(folder.replace(self.exp_name, "")) + 1
            except:
                continue

        return exp_number
    