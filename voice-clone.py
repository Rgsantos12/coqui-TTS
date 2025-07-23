import torch
from TTS.api import TTS

# Check if CUDA is installed
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Print available TTS models
view_models = input("View models? [y/n]\n")
if view_models == "y":
    tts_manager = TTS().list_models()
    all_models = tts_manager.list_models()
    print("TTS models:\n", all_models, "\n", sep = "")

# Prompt model selection
# model = input("Enter model:\n")
# for example, tts_models/pt/cv/vits 

model = r"C:\Users\Utilizador\AppData\Local\tts\tts_models--pt--cv--vits"

# Example voice cloning with selected model
tts = TTS((model), progress_bar=True).to(device)
tts.tts_to_file("Olá pessoal sou o mister nikki e cá estamos para mais um vídeo.", speaker_wav="train-audio.wav"
                , file_path="data/out/output.wav")