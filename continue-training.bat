@echo off
set CUDA_VISIBLE_DEVICES=0
python train_vits_tts_phonemes_ptpt.py --continue_path "models/fine-tuned/one-voice/vits_ssdpt-eumember_96976-July-24-2025_06+25PM-1df082d3" 