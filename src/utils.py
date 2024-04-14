import os
import torch
import wave
import numpy as np
import torchaudio
import torch.nn as nn
from pydub import AudioSegment 
from torchaudio.transforms import Resample
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path


def format_transform(source_folder: str) -> None:
    """
    Change the format of audio from raw to wav.
    There are several folders under a source folder, 
    the audio with format of raw are in these folders.

    Args:
        source_folder: The path of source folder with raw format of audio.   
    """
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.raw'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as raw_file:
                    raw_data = raw_file.read()                                             
                raw_array = np.frombuffer(raw_data, dtype=np.int16)
                sample_width = 2
                frame_rate = 48000  
                wave_file = wave.open(os.path.join(root, file.replace('.raw', '.wav')), 'w')
                wave_file.setnchannels(1)
                wave_file.setsampwidth(sample_width)
                wave_file.setframerate(frame_rate)
                wave_file.writeframes(raw_array.tobytes())
                wave_file.close()
                print(f'Converted {file} to .wav format.')



def Slice_Audio(segment_length: int, input_folder: str, output_folder: str) -> None:
    """
    Preprocess the raw data.
    Slice your audio into a certain length.

    Args:
        segment_length: The cutting time(ms) of each slice of your audio.
        input_folder: Path of your original audio data.
        output_folder: Path of your audio data after sliced.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  
            input_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_wav(input_path)
            num_segments = len(audio) // segment_length
            for i in range(num_segments):
                start_time = i * segment_length
                end_time = (i + 1) * segment_length
                segment = audio[start_time:end_time]
                output_filename = f"segment{i+1}_{filename}"
                output_path = os.path.join(output_folder, output_filename)
                segment.export(output_path, format="wav")



def stereo_to_mono(input_folder: str, output_folder: str) -> None:
    """
    Transform stereo to mono.

    Args:
        input_folder: The path of input_folder with some stereo or mono audio in it.
        output_folder: The path of output_folder.There will be only mono audio in it.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            waveform, sample_rate = torchaudio.load(input_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            torchaudio.save(output_path, waveform, sample_rate)



def resample_to_16k(input_folder: str, output_folder: str):
    """
    Resample your audio to sample rate of 16000Hz.

    Args:
        input_folder: The path of input_folder with some audio in various sample rate.
        output_folder: The path of output_folder.There will be only audio of sample rate of 16000.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    resampler = Resample(orig_freq=48000, new_freq=16000)
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            waveform, _ = torchaudio.load(input_path)
            resampled_waveform = resampler(waveform)
            torchaudio.save(output_path, resampled_waveform, 16000)



class MonoToColor(nn.Module):
    
    def __init__(self, num_channels=3):
        super(MonoToColor, self).__init__()
        self.num_channels = num_channels
        
    def forward(self, x):
        return x.repeat(self.num_channels, 1, 1)
    
    
    
def split_dataset(dataset, train_radio, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    train_size = int(train_radio * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset=dataset, lengths=[train_size, test_size])
    return train_data, test_data



def plot_loss_curves(training_results: dict[str, list[float]]):
    """
    Plots training curves of a results dictionary.
    
    Args:
        training_results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...], "train_acc": [...],
             "test_loss": [...], "test_acc": [...]}
    """
    Train_Loss = training_results['train_loss']
    Test_Loss = training_results['test_loss']
    epochs = range(len(training_results['train_loss']))
    plt.figure(figsize=(18, 12))
    plt.plot(epochs, Train_Loss, label='Train_Loss', linewidth=2)
    plt.plot(epochs, Test_Loss, label='Val_Loss', linewidth=2)
    plt.title("Loss Curves", fontsize=25)
    plt.xlabel("Epochs", fontsize=20)
    plt.gca().xaxis.set_major_locator(MultipleLocator(4))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.show()



def plot_acc_curves(training_results: dict[str, list[float]]):
    
    Train_Accuracy = training_results['train_acc']
    Test_Accuracy = training_results['test_acc']
    epochs = range(len(training_results['train_loss']))
    plt.figure(figsize=(18, 12))
    plt.plot(epochs, Train_Accuracy, label='Train_Acc', linewidth=2)
    plt.plot(epochs, Test_Accuracy, label='Val_Acc', linewidth=2)
    plt.title("Acc Curves", fontsize=25)
    plt.xlabel("Epochs", fontsize=20)
    plt.gca().xaxis.set_major_locator(MultipleLocator(4))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.show()




def save_model(model, target_dir, model_name):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

