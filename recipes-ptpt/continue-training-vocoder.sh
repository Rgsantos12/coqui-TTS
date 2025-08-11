#!/bin/bash

# This script is used to continue training a vocoder in a distributed manner.
# Replace <gpus> with the appropriate GPU device IDs for your setup.
# Use this command, to find the number of the available gpus: 
 # - nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv
# Example: CUDA_VISIBLE_DEVICES="0,1,2,3" for 4 GPUs.

# Also, replace <output_path> with the path to the output directory where the training state is saved. 

# Source: https://github.com/coqui-ai/TTS/discussions/646
CUDA_VISIBLE_DEVICES="<gpus>" python -m trainer.distribute --script TTS/bin/train_vocoder.py --continue_path "<output_path>" 