import numpy as np
from pytorch_msssim import MS_SSIM
from torch import from_numpy
import tqdm

from evaluate import Evaluator


def main():
    features = np.load("../features.npz")
    targets = np.load("../targets.npz")

    criterion = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=1)
    evaluator = Evaluator()

    scores = [
        criterion(
            from_numpy(evaluator.predict(*datum)).view(24, 64, 64).unsqueeze(dim=1),
            from_numpy(target).view(24, 64, 64).unsqueeze(dim=1),
        ).item()
        for *datum, target in tqdm.tqdm(
            zip(features["osgb"], features["data"], targets["data"]),
            total=len(features["data"]),
        )
    ]

    print(f"Score: {np.mean(scores)} ({np.std(scores)})")


if __name__ == "__main__":
    main()
