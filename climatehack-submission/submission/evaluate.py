import numpy as np
import torch
import cv2

from climatehack import BaseEvaluator

import sys

sys.path.append("./dgmr-oneshot")
import dgmr

DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN_PIXEL = 240.3414
_STD_PIXEL = 146.52366


def transform(x):
    return (x - _MEAN_PIXEL) / _STD_PIXEL


def inv_transform(x):
    return (x * _STD_PIXEL) + _MEAN_PIXEL


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return res


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.
        In this case, it loads the trained model (in evaluation mode)."""
        ccs = dgmr.common.ContextConditioningStack(
            input_channels=1,
            conv_type="standard",
            output_channels=160,
        )

        sampler = dgmr.generators.Sampler(
            forecast_steps=24,
            latent_channels=96,
            context_channels=160,
            output_channels=1,
        )
        model = dgmr.generators.Generator(ccs, sampler)
        model.load_state_dict(torch.load("weights/model.pt", map_location=DEVICE))
        self.model = model.to(DEVICE)
        self.model.train()
        # print("DOING TRAIN MODE")

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.
        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (bs, 12, 128, 128)
        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (bs, 24, 64, 64)
        """
        # prediction_opt_flow = self._predict_opt_flow(data)
        prediction_dgmr = self._predict_dgmr(coordinates, data)
        # copy the opt flow predictions in
        # prediction_dgmr[:, : prediction_opt_flow.shape[1]] = prediction_opt_flow
        prediction = prediction_dgmr
        return prediction

    def _predict_dgmr(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        bs = data.shape[0]
        data = data[:, -4:]
        data = torch.FloatTensor(transform(data)).float().to(DEVICE)
        # add a satellite dimension
        data = torch.unsqueeze(data, dim=2)
        # make a batch to help with norm
        # data = torch.cat([data, self.default_batch], dim=0)
        with torch.no_grad():
            prediction = self.model(data)
        # remove the satellite dimension and grab the inner 64x64
        prediction = inv_transform(prediction[:, :, 0, 32:96, 32:96])
        if prediction.device == "cpu":
            prediction = prediction.numpy()
        else:
            prediction = prediction.detach().cpu().numpy()
        return prediction

    def _predict_opt_flow(self, data: np.ndarray) -> np.ndarray:
        bs = data.shape[0]
        forecast = 1
        prediction = np.zeros((bs, forecast, 64, 64), dtype=np.float32)
        test_params = {
            "pyr_scale": 0.5,
            "levels": 2,
            "winsize": 40,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 0.7,
        }
        for i in range(bs):
            sample = data[i]
            cur = sample[-1].astype(np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                prev=sample[-2],
                next=sample[-1],
                flow=None,
                **test_params,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            )
            for j in range(forecast):
                cur = warp_flow(cur, flow)
                prediction[i, j] = cur[32:96, 32:96]
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
