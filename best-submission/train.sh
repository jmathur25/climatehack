#!/bin/bash
CUDA_VISIBLE_DEVICES=0 conda run -n climatehack --no-capture-output --live-stream python /home/iyaja/Git/climatehack/train_unet.py
