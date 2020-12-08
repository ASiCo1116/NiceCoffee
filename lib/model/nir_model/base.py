import torch

from ..base import DeepLearningModule

#--------------------------------------------------------------------------------
# base.py contain the NIR base model class object
#--------------------------------------------------------------------------------


__all__ = ['NIRDeepLearningModule']


class NIRDeepLearningModule(DeepLearningModule):
    def __init__(self):
        super(NIRDeepLearningModule, self).__init__()

        self.is_nir_model = True

    def predict(self, spectrum):
        output = super(NIRDeepLearningModule, self).predict(spectrum)
        output = (torch.sigmoid(output) > 0.5).float().numpy()

        return output


