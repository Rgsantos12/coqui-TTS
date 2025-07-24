import os
import torch
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

def main():
    print(f"Check cuda availability: {torch.cuda.is_available()}")

    print("Starting training script...")
    output_path = "models/fine-tuned/one-voice"

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.txt", path=os.path.join("data", "in-one-speaker", "EUmember_96976")
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_ssdpt-eumember_96976",
        # use_speaker_embedding=True,
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        save_step=1000,
        save_best_after=1000,
        save_checkpoints=True,
        save_n_checkpoints=4,
        text_cleaner="portuguese_cleaners", #multilingual_cleaners
        use_phonemes=False, #True <- helps in convergence, for now its turned off
        phoneme_language="pt", #None
        # phonemizer="multi_phonemizer", # Use the custom phonemizer (espeak pt)
        phoneme_cache_path=os.path.join(output_path, "portuguese_cleaners"),
        compute_input_seq_cache=True,
        print_step=5,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        test_sentences=[
            [
                "Demorei muito tempo para criar esta voz",
                "EUmember_96976",
                None,
                "pt",
            ],
            [
                "Frase de teste para verificar a qualidade da voz",
                "EUmember_96976",
                None,
                "pt",
            ],
        ],
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