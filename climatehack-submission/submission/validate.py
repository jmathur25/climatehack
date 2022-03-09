import numpy as np
from pytorch_msssim import MS_SSIM
from torch import from_numpy
import torch
import tqdm

from evaluate import Evaluator


def main():
    features = np.load("../features.npz")
    targets = np.load("../targets.npz")

    criterion = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=1)
    evaluator = Evaluator()

    batch_size = 16
    split_num = len(features["data"]) // batch_size
    osgb_splits = np.array_split(features["osgb"], split_num, axis=0)
    data_splits = np.array_split(features["data"], split_num, axis=0)
    targets_splits = np.array_split(targets["data"], split_num, axis=0)

    pbar = tqdm.tqdm(
        zip(osgb_splits, data_splits, targets_splits),
        total=len(data_splits),
    )

    scores = []
    for (osgb, data, target) in pbar:
        bs = len(data)
        preds = from_numpy(evaluator.predict(osgb, data))
        trgs = from_numpy(target)
        # this and the batch indexing essentially makes the 24 timesteps the batch
        preds = torch.unsqueeze(preds, dim=2)
        trgs = torch.unsqueeze(trgs, dim=2)

        for i in range(bs):
            # grab the current batch
            score = criterion(preds[i], trgs[i]).item()
            scores.append(score)
        pbar.set_description(f"Avg: {np.mean(scores)}")

    # scores = [
    #     criterion(
    #         from_numpy(evaluator.predict(*datum)).view(24, 64, 64).unsqueeze(dim=1),
    #         from_numpy(target).view(24, 64, 64).unsqueeze(dim=1),
    #     ).item()
    #     for *datum, target in tqdm.tqdm(
    #         zip(features["osgb"], features["data"], targets["data"]),
    #         total=len(features["data"]),
    #     )
    # ]

    print(f"Score: {np.mean(scores)} ({np.std(scores)})")


if __name__ == "__main__":
    main()
