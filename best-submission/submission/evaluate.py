import numpy as np
import torch
import torch.nn as nn
from functools import partial

from climatehack import BaseEvaluator

# from model import Model

from fastai.vision.all import create_unet_model, models
from fastai.vision.models.xresnet import *
from fastai.layers import Mish


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        arch = partial(xse_resnext50_deeper, act_cls=Mish, sa=True)
        self.model = create_unet_model(
            arch=arch,
            img_size=(128, 128),
            n_out=24,
            pretrained=False,
            n_in=12,
            self_attention=True,
        ).cpu()

        self.model.load_state_dict(
            torch.load("xse_resnext50_deeper.pth", map_location="cpu")
        )
        self.model.eval()

        self.arcnn = ARCNN(weights).to("cpu").eval()

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)

        with torch.no_grad():

            inp = torch.from_numpy(data).unsqueeze(0)
            preds = self.model(inp)
            assert prediction.shape == (24, 64, 64)

            return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
