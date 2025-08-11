import os
import shutil
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

# Appends speaker information from europarl dataset to the transcript file
def add_speakers_from_europarl_to_transcript(transcript_file, info_dir):
    """
    Adds speaker information to the transcript file by matching audio filenames with Europarl dataset.
    (Useful for fine-tuning a tts model with one speaker only)

    Args:
        transcript_file (str): Path to the transcript TSV file
        info_dir (str): Path to the Europarl info directory containing speeches.lst and speakers.lst
    """
    # Main file
    file = transcript_file
    for folder in ["train", "dev", "test"]:

        # Read speeches.lst and speakers.lst
        speeches_file = os.path.join(info_dir, folder, "speeches.lst")
        speakers_file = os.path.join(info_dir, folder, "speakers.lst")
        
        # Create mappings from speeches.lst to indices
        speech_to_index = {}
        with open(speeches_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                speech_to_index[idx] = line.strip()
        # print(speech_to_index) #result
        # print(f"Loaded {len(speech_to_index)} speeches from {speeches_file}")

        # Read speakers list and retrieve speaker names
        speakers = []
        with open(speakers_file, 'r', encoding='utf-8') as f:
            speakers = [line.strip().split('/')[-1] for line in f]
        # print(speakers)
        
        # Read and process transcript file
        temp_output = file + '.tmp'
        with open(file, 'r', encoding='utf-8') as fin, \
                open(temp_output, 'w', encoding='utf-8') as fout:
            
            # Write header
            header = fin.readline().strip()
            fout.write(f"{header}\tspeaker\n")
            
            # Process each line of the tsv file
            for line in fin:
                if "train" in folder:
                    audio, text = line.strip().split('\t')
                    speaker = "Unknown" # Default speaker
                else:
                    audio, text, speaker = line.strip().split('\t')
                    # print(line.strip().split('\t'))

                # Search for the id in the mappings, in the tsv file
                for speech_id, speech in speech_to_index.items():
                    if speech in audio:
                        # If the speech ID is found in the audio filename
                        # print(f"Found speech ID {speech_id} for audio {audio}")
                        # Retrieve the index position of the speech which corresponds to the speaker name
                        speaker = speakers[speech_id]

                fout.write(f"{audio}\t{text}\t{speaker}\n")
        file = temp_output

    # Replace original file with new version
    os.replace(file, "data/real_transcriptV2.tsv")

    # Remove temporary file if it exists
    for file in os.listdir('data'):
        file_path = os.path.join('file', file)
        if file_path.endswith('.tmp'):
            os.remove(file_path)


# Appends speaker information from all datasets besides europarl to the transcript file
def add_speakers_from_other_to_transcript(transcript_file):
    # Main file
    file = transcript_file

    # Read and process transcript file
    temp_output = file + '.tmp'
    with open(file, 'r', encoding='utf-8') as fin, \
            open(temp_output, 'w', encoding='utf-8') as fout:
        
        # Write header
        header = fin.readline().strip()
        fout.write(f"{header}\n")
        
        # Process each line of the tsv file
        for line in fin:
            audio, text, speaker = line.strip().split('\t')
            # print(line.strip().split('\t'))
            if speaker == "Unknown":
                if "mozilla" in audio:
                    speaker = f"MCV_{audio.split('_')[-1].replace('.wav', '')}"  # Extract speaker ID from filename
                elif "vxforge" in audio:
                    speaker = f"VXF_{audio.split('_')[1]}"
                else:
                    print("Unknown speaker for audio:", audio)
                    break
                fout.write(f"{audio}\t{text}\t{speaker}\n")
                continue

            elif "EUmember" in speaker:
                # Skip europarl speakers
                fout.write(f"{audio}\t{text}\t{speaker}\n")
                continue
            elif "Speaker" in speaker:
                # Skip individual speakers of europarl
                fout.write(f"{audio}\t{text}\t{speaker}\n")
                continue
            else:
                print("Error: ", audio)
                break
        file = temp_output

    # Replace original file with new version
    os.replace(file, "data/real_transcriptV3.tsv")


# Used to fetch a single speaker's audio files from a dataset directory
def get_speaker_audio_and_txt(data_dir, transcript_dir, speaker_id=None):
    """"
    Fetches audio files from a dataset directory and saves their transcripts to a specified output directory.
    Args:
        data_dir (str): Path to the dataset directory containing audio files.
        output_transcript_dir (str): Path to the directory where transcripts will be saved.
    """
    if speaker_id is not None:
        print(f"Preparing dataset for speaker: {speaker_id}")
        path = f"data/in-one-speaker/{speaker_id}"
        one_speak_dir = os.path.join(path, "wavs")
        os.makedirs(one_speak_dir, exist_ok=True)
    else:
        print(f"Preparing dataset for multiple speakers")
        path = f"data/in-multi-speakers"
        mult_speak_dir = os.path.join(path, "wavs")
        os.makedirs(mult_speak_dir, exist_ok=True)

    # Obtain all audio files of a given speaker from tsv file
    file = os.path.join(path, "metadata.txt")
    with open(transcript_dir, 'r', encoding='utf-8') as fin, \
            open(file, 'w', encoding='utf-8') as fout:
        next(fin) # Skip header
        # Process each line
        for i, line in enumerate(fin, start=1):
            audio, text, speaker = line.strip().split('\t')

            if speaker == speaker_id:
                fout.write(f"real_{i}_{audio.replace('.wav', '')}|{text}|{text}\n") #Same text two times for now
                shutil.copyfile(os.path.join(data_dir, f"real_{i}_{audio}"), 
                            os.path.join(path, "wavs", f"real_{i}_{audio}"))
            elif speaker_id is None:
                fout.write(f"real_{i}_{audio}|{text}|{speaker}\n") #Also text not normalized for now
                shutil.copyfile(os.path.join(data_dir, f"real_{i}_{audio}"), 
                            os.path.join(path, "wavs", f"real_{i}_{audio}"))

    # print(audio_files)
    # print(f"Found {len(audio_files)} audio files for speaker {speaker_id}")


# custom formatter implementation for multiple speakers
# def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
#     """Assumes each line as ```<filename>|<transcription>|<speaker>```
#     """
#     txt_file = os.path.join(root_path, manifest_file)
#     items = []
#     with open(txt_file, "r", encoding="utf-8") as ttf:
#         for line in ttf:
#             cols = line.split("|")
#             wav_file = os.path.join(root_path, "wavs", cols[0])
#             if not os.path.exists(wav_file):
#                 print(f"Audio file {wav_file} does not exist, skipping line: {line.strip()}")
#                 continue
#             text = cols[1]
#             speaker_id = cols[2]
#             items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_id, "root_path": root_path})
#     return items


if __name__ == "__main__":
    """
    Here are the speakers on the real part of the dataset ssd pt-pt:
    Speakers from eparl include:
    - EUmember_96706
    - EUmember_28310
    - EUmember_96820
    - EUmember_4466
    Other speakers include: #1 only \ Less audio files
    - MCV_38377514
    - VXF_anonymous-20120523-oiu
    ...
    """
    transcript_old_dir = "data/real_transcript.tsv"
    transcript_dir = "data/real_transcriptV2.tsv"  # Replace with your dataset path
    info_dir = r"C:\Users\Utilizador\OneDrive\Documentos\Europarl-st v1.1\pt\en"
    data_dir = r"E:\Datasets\synthetic-speech-detection-dataset-ptpt\real"
    
    # Add speakers from europarl
    # add_speakers_from_europarl_to_transcript(transcript_old_dir, info_dir) Done

    # Add the rest of speakers
    # add_speakers_from_other_to_transcript(transcript_dir)
    
    transcript_v3_dir = "data/real_transcriptV3.tsv" #latest version

    speaker_id = "EUmember_96706"  # Example speaker ID
    get_speaker_audio_and_txt(data_dir, transcript_v3_dir, speaker_id=speaker_id) 
    # get_speaker_audio_and_txt(data_dir, transcript_v3_dir, speaker_id=None) # For multiple speakers

    # For single speaker -> Ljspeech format:
    # Metadata is provided in transcripts.csv. -> im providing a txt but its the same 
    # This file consists of one record per line, delimited by the pipe character (0x7c). The fields are:
    #  - ID: this is the name of the corresponding .wav file
    #  - Transcription: words spoken by the reader (UTF-8)
    #  - Normalized Transcription: transcription with numbers, ordinals, and monetary units expanded into full words (UTF-8).
    # Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 22050 Hz. 

    # For multiple speakers -> custom format
    # result = formatter("data/in-multi-speakers", "metadata.txt")  
    # print(result[0]) #Example

    ############## DONE IN train ###########################
    # Start finetuning a model
    # https://docs.coqui.ai/en/latest/training_a_model.html
    # https://docs.coqui.ai/en/latest/finetuning.html

    # Load samples from the dataset
    # dataset_config = BaseDatasetConfig(
    #     formatter=formatter, meta_file_train="metadata.txt", language="pt-pt", path="data/in/"
    # )

    # # Load training samples
    # train_samples, eval_samples = load_tts_samples(
    #     dataset_config,
    #     eval_split=True,
    #     formatter=formatter  # Pass the function here
    # )
    # print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")