# climatehack

`climatehack-submission/` is for submitting. It has a `submit.sh` that you can use to submit the current model. There is a `submission` folder that is zipped and it should run in a self-contained way. You can see I’ve included a copy of the DGMR repo there. Inside `submission` there is a `validate.py`. You can execute this file with `python validate.py` to see if the submission code properly runs.

`common/` has basic utilities
`data/` has the 300d sample and the full dataset
`models/` has the proof of concept of DGMR but is otherwise useless
`weights/` is where all weights are being saved

In the repo root we have a bunch of files. Their names mostly explain what they do. The most important note might be that `train_dgmr_forecast=24_full.ipynb` seems to be weird… it loads data from disk but I found it was decreasing performance. I might have some bug with my disk data loader. The in_memory script is what I was working through and seems to work. Right now it is slightly modded cause I was trying to make it backprop MS-SSIM and it’s failing, but you can go thru and see for yourself.  
