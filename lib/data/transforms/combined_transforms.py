from .base import TransformFunctionObject
from .nir_preprocess import NIRPreprocess
from .gcms_preprocess import GCMSPreprocess
from .transforms import Compose

#--------------------------------------------------------------------------------
# combined_transforms.py containes the object which can combine NIRPreprocess and
# Compose.
#--------------------------------------------------------------------------------


__all__ = ['CombinedTransforms']


class CombinedTransforms(TransformFunctionObject):
    def __init__(self, transforms = None, preprocess = None):
        if transforms is not None:
            if not isinstance(transforms, Compose):
                raise TypeError('transforms in CombinedTransforms must be the lib.data.transforms.Compose object')

        self.transforms = transforms

        if preprocess is not None:
            if not isinstance(preprocess, (GCMSPreprocess, NIRPreprocess)):
                raise TypeError('preprocess in CombinedTransforms must be the NIRPreprocess or GCMSPreprocess object.')

        self.preprocess = preprocess

    def __call__(self, spectrum):
        if self.preprocess is not None:
            spectrum = self.preprocess(spectrum)

        if self.transforms is not None:
            spectrum = self.transforms(spectrum)

        return spectrum

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.preprocess is not None:
            format_string += 'preprocess:\n'
            format_string += self.preprocess.__repr__()

        if self.transforms is not None:
            format_string += 'transforms:\n'
            format_string += self.transforms.__repr__()

        format_string += ')'
        return format_string




