import numpy as np
import torch

from climatehack import BaseEvaluator
from model import Model
import cv2


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""
        pass
        # self.model = Model()
        # self.model.load_state_dict(torch.load("model.pt"))
        # self.model.eval()

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

        prev = data[-2]
        cur = data[-1]
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev,
            next=cur,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=10,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )

        prediction = np.zeros((24, 64, 64), dtype=np.float32)
        cutoff = 6
        for j in range(1, 25):
            yhat = warp_flow(cur, flow * j)
            if j <= cutoff:
                # if the multiplier is not too bad, do this:
                # apply the flow `i` iterations
                yhat = warp_flow(cur, flow * j)[32 : 32 + 64, 32 : 32 + 64]
            else:
                # otherwise, stick with the last optical flow prediction
                yhat = cur[32 : 32 + 64, 32 : 32 + 64]  # prediction[cutoff - 2]

            prediction[j - 2, :, :] = yhat

        assert prediction.shape == (24, 64, 64)
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
