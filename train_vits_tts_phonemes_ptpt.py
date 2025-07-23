import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# custom formatter implementation
# def custom_formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
#     """Assumes each line as ```<filename>|<transcription>```
#     """
#     txt_file = os.path.join(root_path, manifest_file)
#     items = []
#     speaker_name = "EUmember_4466"
#     with open(txt_file, "r", encoding="utf-8") as ttf:
#         for line in ttf:
#             cols = line.split("|")
#             wav_file = os.path.join(root_path, "wavs", cols[0])
#             text = cols[1]
#             items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
#     return items

print("Starting training script...")
output_path = "models"

print("Loading dataset configuration...")
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.txt", language="pt", path="data/in/"
)

print("Loading VitsAudioConfig configuration...")
audio_config = VitsAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

# vitsArgs = VitsArgs(
#     # use_language_embedding=True,
#     embedded_language_dim=4,
#     # use_speaker_embedding=True, # only train on one now
#     use_sdp=False,
# )

config = VitsConfig(
    audio=audio_config,
    run_name="vits_ssdpt-finetune-eumember_4466",
    # use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10,
    text_cleaner="portuguese_cleaners", #multilingual_cleaners
    use_phonemes=True,
    phoneme_language="pt", #None
    # phonemizer="multi_phonemizer", # Use the custom phonemizer (espeak pt)
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=dataset_config,
    cudnn_benchmark=False,
)

# force the convertion of the custom characters to a config attribute
# config.from_dict(config.to_dict())

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
# speaker_manager = SpeakerManager()
# speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
# config.model_args.num_speakers = speaker_manager.num_speakers

# init lang manager for multi-lang training
# language_manager = LanguageManager(config=config)
# config.model_args.num_languages = language_manager.num_languages

# # init model
model = Vits(config, ap, tokenizer, speaker_manager=None) #speaker_manager, language_manager

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()

print("Training completed successfully!")
