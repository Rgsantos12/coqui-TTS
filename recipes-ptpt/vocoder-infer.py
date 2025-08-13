import os
import torch
import importlib
import re
import glob

from TTS.config import load_config
from TTS.vocoder.models import setup_model
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.datasets.gan_dataset import GANDataset


"""
This script is used to infer audio using a GAN vocoder model.
"""

# Check if CUDA is installed
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Prompt model selection (download the model)
# model = input("Enter model:\n")

# Choose the vocoder
base_path = r"C:\Users\Utilizador\AppData\Local\tts\vocoder_models--en--ljspeech--hifigan_v2"
vocoder_path = os.path.join(base_path, "model_file.pth")
vocoder_config_path = os.path.join(base_path, "config.json")

# Setup the vocoder
config = load_config(vocoder_config_path)
vocoder = setup_model(config)

# Restore its parameters
is_eval = True
vocoder.load_checkpoint(config, vocoder_path, eval=is_eval)
vocoder.to(device)

# Initialize the audio processor
ap = AudioProcessor.init_from_config(config)

# Define audio samples path
base_path = "../data/test"
samples = glob.glob(
    os.path.join(base_path, "*.wav")
)

for i in range(len(samples)):
    filename = str(os.path.basename(samples[i])).rsplit(".")[0]
    print(f"Processing file: {filename}\n")

    mel = ap.melspectrogram(ap.load_wav(samples[i], sr=config.audio["sample_rate"]))
    mel_torch = torch.from_numpy(mel).to(device)

    # print(f"Mel spectrogram shape: {mel_torch.shape}")
    neural_vocoder_wav = vocoder.inference(mel_torch) # method from GAN API
    # print(f"Output vocoder fake audio shape: {neural_vocoder_wav.squeeze(0).shape}")
    neural_vocoder_wav = neural_vocoder_wav.squeeze(0).cpu().numpy()

    # Define output paths
    output_path = f"{base_path}/output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wav_gf_path = os.path.join(output_path, f"{filename}_gl.wav")
    wav_neural_path = os.path.join(output_path, f"{filename}_{config.model.lower()}_neural.wav")
    
    gf_wav = ap.inv_melspectrogram(mel) # uses Griffin-Lim vocoder
    ap.save_wav(gf_wav, wav_gf_path, sr=config.audio["sample_rate"])
    print("Audio from griffin-lim vocoder saved at:", wav_gf_path)

    ap.save_wav(neural_vocoder_wav, wav_neural_path, sr=config.audio["sample_rate"])
    print(f"Audio from {config.model.lower()} saved at:", wav_neural_path)


## More efficient for large data (not needed for now)
# dataset = GANDataset(
#     ap=ap,
#     items=samples,
#     seq_len=config.seq_len,
#     hop_len=ap.hop_length,
#     pad_short=config.pad_short,
#     conv_pad=config.conv_pad,
#     return_pairs=config.diff_samples_for_G_and_D if "diff_samples_for_G_and_D" in config else False,
#     is_training=not is_eval,
#     return_segments=not is_eval,
#     use_noise_augment=config.use_noise_augment,
#     use_cache=config.use_cache,
# ) # vocoder.get_data_loader()

# sampler = None
# loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=1 if is_eval else config.batch_size,
#     shuffle=False,
#     drop_last=False,
#     sampler=sampler,
#     num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers, 
#     pin_memory=False,
# ) 
