import numpy as np
import torch
import cv2

from climatehack import BaseEvaluator

import sys

sys.path.append("./dgmr-mod")
import dgmr


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cpu")

# _MEDIAN_PIXEL = 212.0
# _IQR = 213.0
_MEAN_PIXEL = 240.3414
_STD_PIXEL = 146.52366


def transform(x):
    # return np.tanh((x - _MEDIAN_PIXEL) / _IQR)
    return (x - _MEAN_PIXEL) / _STD_PIXEL


def inv_transform(x):
    # return torch.atanh(x) * _IQR + _MEDIAN_PIXEL
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

        # model = dgmr.DGMR(
        #     forecast_steps=24,
        #     input_channels=1,
        #     output_shape=128,
        #     latent_channels=384,
        #     context_channels=192,
        #     num_samples=3,
        # )
        # ccs = dgmr.common.ContextConditioningStack(
        #     input_channels=1,
        #     conv_type="standard",
        #     output_channels=192,
        # )
        # lcs = dgmr.common.LatentConditioningStack(
        #     shape=(8 * 1, 128 // 32, 128 // 32),
        #     output_channels=384,
        # )
        # sampler = dgmr.generators.Sampler(
        #     forecast_steps=24,
        #     latent_channels=384,
        #     context_channels=192,
        # )
        # ccs = dgmr.common.ContextConditioningStack(
        #     input_channels=1,
        #     conv_type="standard",
        #     output_channels=96,
        # )

        # lcs = dgmr.common.LatentConditioningStack(
        #     shape=(4 * 1, 128 // 32, 128 // 32),
        #     output_channels=192,
        # )

        # sampler = dgmr.generators.Sampler(
        #     forecast_steps=24,
        #     latent_channels=192,
        #     context_channels=96,
        # )
        self.default_batch = torch.load("weights/default_batch.pt", map_location=DEVICE)
        ccs = dgmr.common.ContextConditioningStack(
            input_channels=1,
            conv_type="standard",
            output_channels=144,  # 96
        )

        lcs = dgmr.common.LatentConditioningStack(
            shape=(4 * 1, 128 // 32, 128 // 32),
            output_channels=288,  # 192
        )

        sampler = dgmr.generators.Sampler(
            forecast_steps=24,
            latent_channels=288,  # 192
            context_channels=144,  # 96
        )
        model = dgmr.generators.Generator(ccs, lcs, sampler)
        model.load_state_dict(torch.load("weights/model.pt", map_location=DEVICE))
        self.model = model.to(DEVICE)
        self.model.train()
        # print("DOING TRAIN MODE")

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

        prediction_opt_flow = self._predict_opt_flow(data)
        prediction_dgmr = self._predict_dgmr(coordinates, data)
        # copy the opt flow predictions in
        prediction_dgmr[: len(prediction_opt_flow)] = prediction_opt_flow

        prediction = prediction_dgmr
        assert prediction.shape == (24, 64, 64)
        return prediction

    def _predict_dgmr(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        data = data[-4:]
        data = torch.FloatTensor(transform(data)).float().to(DEVICE)
        data = torch.unsqueeze(data, dim=0)
        data = torch.unsqueeze(data, dim=2)
        # make a batch to help with norm
        data = torch.cat([data, self.default_batch], dim=0)
        with torch.no_grad():
            prediction = self.model(data)
        # grab first entry cause that's the real prediction
        prediction = prediction[:1]
        prediction = inv_transform(prediction[:, :, :, 32:96, 32:96])
        prediction = np.squeeze(prediction.numpy())
        # prediction = np.squeeze(prediction.detach().cpu().numpy())

        return prediction

    def _predict_opt_flow(self, data: np.ndarray) -> np.ndarray:
        test_params = {
            "pyr_scale": 0.5,
            "levels": 2,
            "winsize": 40,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 0.7,
        }
        forecast = 3
        prediction = np.zeros((forecast, 64, 64), dtype=np.float32)
        cur = data[-1].astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            prev=data[-2],
            next=data[-1],
            flow=None,
            **test_params,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        for i in range(forecast):
            cur = warp_flow(cur, flow)
            prediction[i] = cur[32:96, 32:96]
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
