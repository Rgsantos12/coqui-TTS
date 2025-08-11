import os
import torch

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

def main():
    print(f"Check cuda availability: {torch.cuda.is_available()}")

    # dir_path = os.path.dirname(os.path.abspath(__file__))
    output_path = "../../models/trained/one-voice/hifigan/"

    config = HifiganConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=5,
        epochs=1000,
        save_on_interrupt=True,
        save_step=10, # Number of training steps expected to save traning stats and checkpoints.
        save_checkpoints=True, # If true, it saves checkpoints per "save_step"
        seq_len=8192,
        pad_short=2000,
        use_noise_augment=True,
        eval_split_size=10,
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        lr_gen=1e-4,
        lr_disc=1e-4,
        data_path="../../data/in-one-speaker/EUmember_96706/wavs/",
        output_path=output_path
    )

    # init audio processor
    ap = AudioProcessor(**config.audio.to_dict())

    # load training samples
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

    # init model
    model = GAN(config, ap)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()

if __name__ == "__main__":
    main()