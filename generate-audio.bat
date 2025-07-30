@echo off
set CUDA_VISIBLE_DEVICES=0

:: Define paths
set MODEL_PATH=C:\Users\Utilizador\OneDrive\Documentos\synthetic-audio-gen\TTS\models\fine-tuned\one-voice\vits_ssdpt-EUmember_4466-July-26-2025_02+11AM-1df082d3\checkpoint_5275.pth
set CONFIG_PATH=C:\Users\Utilizador\OneDrive\Documentos\synthetic-audio-gen\TTS\models\fine-tuned\one-voice\vits_ssdpt-EUmember_4466-July-26-2025_02+11AM-1df082d3\config.json
set OUTPUT_PATH=data\out\output.wav

:: Run TTS command with variables
tts --text "Olá Pessoal sou eu, como estão" --model_path "%MODEL_PATH%" --config_path "%CONFIG_PATH%" --out_path "%OUTPUT_PATH%"