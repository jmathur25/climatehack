import numpy as np
import torch

from climatehack import BaseEvaluator
import metnet
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MAX_PIXEL = 1023
_MEAN = 0.1787


def transform(x):
    return (x / _MAX_PIXEL) - _MEAN


def inv_transform(x):
    return (x + _MEAN) * _MAX_PIXEL


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        self.EXPECTED = 24
        self.FORECAST = 10

        model = metnet.MetNet(
            hidden_dim=32,
            forecast_steps=self.FORECAST,  # should be 24 timesteps out
            input_channels=1,  # 12 timeteps in
            output_channels=1,  # 1 data channel in
            sat_channels=1,  # 1 data channel in
            input_size=32,  # =128/4, where 128 is the image dimensions
        )
        self.model = model.to(DEVICE)
        self.model.eval()

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

        data = torch.FloatTensor(transform(data)).to(DEVICE)

        data = data.unsqueeze(data, dim=0)
        data = data.unsqueeze(data, dim=2)
        with torch.no_grad():
            preds = self.model(data)

        # remove the batch and satellite channel dimension from the prediction
        preds = torch.squeeze(preds)
        preds = inv_transform(preds)

        missing = self.EXPECTED - self.FORECAST
        # just do persistence for the missing
        missing_pred = inv_transform(data[:, -1, :, 32:96, 32:96])
        missing_pred = torch.squeeze(missing_pred)
        missing_pred = torch.tile(missing_pred, (missing, 1, 1))
        # stack them together
        prediction = torch.cat([preds, missing_pred], dim=0)

        assert prediction.shape == (24, 64, 64)
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
