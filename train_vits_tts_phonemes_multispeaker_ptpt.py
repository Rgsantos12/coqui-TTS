import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# ======================================== SSD PT-PT ===========================================
# custom formatter implementation
def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument

    """Assumes each line as ```<filename>|<transcription>|<speaker>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            if not os.path.exists(wav_file):
                print(f"Audio file {wav_file} does not exist, skipping line: {line.strip()}")
                continue
            text = cols[1]
            speaker_id = cols[2]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_id, "root_path": root_path})
    return items

output_path = "models/trained"
dataset_config = BaseDatasetConfig(
    formatter=formatter, meta_file_train="metadata.txt", path=os.path.join("data", "in-multi-speakers")
)

audio_config = VitsAudioConfig(
    sample_rate=16000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

vitsArgs = VitsArgs(
    use_speaker_embedding=True,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_ssdptpt_trained_multi_and_phonemes",
    save_on_interrupt=True,
    save_step=10000, # Number of training steps expected to save traning stats and checkpoints.
    save_checkpoints=True, # If true, it saves checkpoints per "save_step"
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="portuguese_cleaners", #portuguese_cleaners (basic (phonemizer does the rest)(text.replace("&", " e "))) #multilingual_cleaners(not needed for pt only)
    use_phonemes=True, #True <- helps in convergence, for now its turned on
    phoneme_language="pt", #None #(pt works too)
    phonemizer="espeak", # Use the custom phonemizer(espeak)  #multi_phonemizer
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    # max_text_len=325,  # change this if you have a larger VRAM than 16GB
    output_path=output_path,
    # datasets=[dataset_config],
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter,  # Use the custom formatter
)
print(f"Train examples: {train_samples[:5]}, Size: {len(train_samples)}\n")
print(f"Eval examples: {eval_samples[:5]}, Size: {len(eval_samples)}\n")

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers
print(f"Number of speakers: {speaker_manager.num_speakers}\n")

# # init model
model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

print("Starting training a model from zero...\n")
trainer.fit()
