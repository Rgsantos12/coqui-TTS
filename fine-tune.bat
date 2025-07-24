@echo off
REM Ask user for the folder containing the model (.pth or.pth.tar file)
set /p MODEL_PATH="Enter the path for the model: " 

REM Set the CUDA device and run the training script with the constructed restore path
set CUDA_VISIBLE_DEVICES=0
python train_vits_tts_phonemes_ptpt.py --restore_path "%MODEL_PATH%"