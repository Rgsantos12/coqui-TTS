import os
import torch
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def main():
    print(f"Check cuda availability: {torch.cuda.is_available()}")

    print("Starting training script...")
    output_path = "models/trained/one-voice"

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.txt", path=os.path.join("data", "in-one-speaker", "EUmember_4466")
    )

    audio_config = VitsAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_ssdpt-eumember_4466",
        # use_speaker_embedding=True,
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        save_on_interrupt=True,
        save_step=10000, # Number of training steps expected to save traning stats and checkpoints.
        save_checkpoints=True, # If true, it saves checkpoints per "save_step"
        text_cleaner="portuguese_cleaners", #portuguese_cleaners(basic (phonemizer does the rest)(text.replace("&", " e "))) #multilingual_cleaners(not needed for pt only)
        use_phonemes=True, #True <- helps in convergence, for now its turned on
        phoneme_language="pt", #None #(pt works too)
        phonemizer="espeak", # Use the custom phonemizer(espeak)  #multi_phonemizer
        phoneme_cache_path=os.path.join(output_path, "phoneme_cleaners"),
        compute_input_seq_cache=True,
        print_step=5,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
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

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

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

if __name__ == '__main__':
    main()