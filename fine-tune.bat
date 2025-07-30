@echo off

REM Set the CUDA device and run the training script with the constructed restore path
set CUDA_VISIBLE_DEVICES=0
python train_vits_tts_pretrained_ptpt.py --config_path  "C:\Users\Utilizador\OneDrive\Documentos\synthetic-audio-gen\TTS\models\fine-tuned\one-voice\config.json" --restore_path "C:\Users\Utilizador\AppData\Local\tts\tts_models--pt--cv--vits\model_file.pth.tar"