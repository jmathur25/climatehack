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


