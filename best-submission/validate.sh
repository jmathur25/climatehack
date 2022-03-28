#!/bin/bash
cd submissionv2
conda run -n climatehack --no-capture-output --live-stream python /home/iyaja/Git/climatehack/submissionv2/validate.py
conda run -n climatehack --no-capture-output --live-stream python /home/iyaja/Git/climatehack/submissionv2/validate_timestep.py
