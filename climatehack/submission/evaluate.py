import numpy as np
import torch

from climatehack import BaseEvaluator
import models
import cv2
import torch


MEAN = 209.0
STD = 60.0
DEVICE = torch.device("cpu")


def _predict(model, srcs):
    batch_size = srcs.shape[0]

    # tensor to store decoder outputs
    outputs = torch.zeros(batch_size, 24, 64, 64).to(DEVICE)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    hidden, cell = model.encoder(srcs)

    # first input to the decoder is the last image in the hour
    # shape: (batch_size, 1, 64, 64)
    input = srcs[:, -1, 32:96, 32:96]
    input = torch.unsqueeze(input, 1)

    for t in range(24):
        output, hidden, cell = model.decoder(input, hidden, cell)
        # images of shape (batch_size, 64, 64)
        output = output.view(batch_size, 64, 64)
        outputs[:, t, :, :] = output

        input = output
        input = torch.unsqueeze(input, 1)

    return outputs


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        INPUT_DIM = 128 * 128
        OUTPUT_DIM = 64 * 64
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 2
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        enc = models.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = models.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        self.model = models.Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        self.model.load_state_dict(
            torch.load("simple_rnn_rand_15.pt", map_location=DEVICE)
        )
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

        data = torch.unsqueeze(torch.FloatTensor((data - MEAN) / STD), 0)
        prediction = _predict(self.model, data)
        prediction = prediction * STD + MEAN
        prediction = prediction[0].detach().numpy()  # only 1 batch

        assert prediction.shape == (24, 64, 64)
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
