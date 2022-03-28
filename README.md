# climatehack


`climatehack-submission/` is for submitting. It has a `submit.sh` that you can use to submit the current model. There is a `submission` folder that is zipped and it should run in a self-contained way. You can see I’ve included a copy of the DGMR repo there. Inside `submission` there is a `validate.py`. You can execute this file with `python validate.py` to see if the submission code properly runs.

`common/` has basic utilities
`data/` has the 300d sample and the full dataset
`models/` has the proof of concept of DGMR but is otherwise useless
`weights/` is where all weights are being saved

In the repo root we have a bunch of files. Their names mostly explain what they do. The most important note might be that `train_dgmr_forecast=24_full.ipynb` seems to be weird… it loads data from disk but I found it was decreasing performance. I might have some bug with my disk data loader. The in_memory script is what I was working through and seems to work. Right now it is slightly modded cause I was trying to make it backprop MS-SSIM and it’s failing, but you can go thru and see for yourself.  

# Setup
```bash
conda env create -f environment.yaml
conda activate climatehack
python -m ipykernel install --user --name=climatehack
```

First, download data by running `data/download_data.ipynb`. Alternatively, you can find preprocessed data files [here](https://drive.google.com/drive/folders/1JkPKjOBtm3dlOl2fRTvaLkSu7KnZsJGw?usp=sharing). We used `train.npz` and `test.npz`. They consist of data temporally cropped from 10am to 4pm UK time across the entire dataset. You could also use `data_good_sun_2020.npz` and `data_good_sun_2021.npz`, which consist of all samples where the sun elevation is at least 10 degrees.


# Best Submission


# Experiments
We conducted several experiments that showed improvements on a strong baseline. The baseline was OpenClimateFix's skillful nowcasting [repo](https://github.com/openclimatefix/skillful_nowcasting), which itself is a reimplementation of Deepmind's precipitation forecasting GAN. This baseline is more-or-less copied to `experiments/dgmr-original`. This baseline will actually train to a score greater than 0.8 on the Climatehack [leaderboard](https://climatehack.ai/compete/leaderboard). We didn't have time to properly test these experiments on top of our best model, but we suspect they would improve results. The experiments are summarized below:

Experiment | Description | Results |
--- | --- | --- |
DCT-Trick | Inspired by [this](https://proceedings.neurips.cc/paper/2018/file/7af6266cc52234b5aa339b16695f7fc4-Paper.pdf), we use the DCT to turn 128x128 -> 64x16x16 and IDCT to turn 64x16x16 -> 128x128. This leads to a shallower network that is autoregressive at fewer spatial resolutions. We believe this is the first time this has been done with UNETs. A fast implementation is in `common/utils.py:create_conv_dct_filter` and `common/utils.py:get_idct_filter`. | 1.8-2x speedup, small <0.005 performance drop |
Denoising | We noticed a lot of blocky artifacts in predictions. These artifacts are reminiscent of JPEG/H.264 compression artifacts. We show a comparison of these artifacts in the [slides](https://docs.google.com/presentation/d/1P_cv3R7gTRXG41wFPXT2lZe9E1GnKqtaJVqe-vsAvL0/edit?usp=sharing). We found a pretrained neural network to fix them. This can definitely be done better, but we show a proof-of-concept. | No performance drop, small visual improvement. The slides have an example. |
CoordConv | Meteorological phenomenon are correlated with geographic coordinates. We add 2 input channels for the geographic coordinates in OSGB form. | +0.0072 MS-SSIM improvement |
Optical Flow | Optical flow does well for the first few timesteps. We add 2 input channels for the optical flow vectors. | +0.0034 MS-SSIM improvement |


