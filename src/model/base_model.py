import torch
import numpy as np
import torch.nn as nn

from . import resnet

USE_CUDA = torch.cuda.is_available()

__all__ = ['CoffeeModel']

class CoffeeModel:
    def __init__(self):
        self.device = "cpu"
        self._setup()

    def _setup(self):
        # self.criterion = nn.MSELoss().to(self.device)
        num_classes = 1
        self.model = resnet.__dict__['resnet18'](num_classes)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.model.to(self.device)

    def predict(self, spectrum):

        self.model.eval()
        spectrum = torch.tensor(spectrum)
        output = np.empty((1, ))

        with torch.no_grad():
            y_hat = self.model(spectrum.float().to(self.device))

        output = np.squeeze(y_hat.to(self.device).numpy())
        return output

    def load_networks(self, model_name):
        
        model = getattr(self, 'model')
        model.load_state_dict(torch.load(model_name))