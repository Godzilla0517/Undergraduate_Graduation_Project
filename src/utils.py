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
    
    
    
def split_dataset(dataset, train_radio, test_ratio, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    test_size = int(test_ratio * len(dataset))
    train_val_size = len(dataset) - test_size
    train_val_data, test_data = random_split(dataset, [train_val_size, test_size])
    train_size = int(train_radio * len(train_val_data))
    val_size = train_val_size - train_size
    train_data, val_data = random_split(train_val_data, [train_size, val_size]) 
    return train_data, val_data, test_data



def plot_loss_curves(training_results: dict[str, list[float]]):
    """
    Plots training curves of a results dictionary.
    
    Args:
        training_results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...], "train_acc": [...],
             "val_loss": [...], "val_acc": [...]}
    """
    Train_Loss = training_results['train_loss']
    Test_Loss = training_results['val_loss']
    epochs = range(len(training_results['train_loss']))
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, Train_Loss, label='Train_Loss', linewidth=2)
    plt.plot(epochs, Test_Loss, label='Validation_Loss', linewidth=2)
    plt.title("Loss Curves", fontsize=25)
    plt.xlabel("Epochs", fontsize=20)
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.025))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.show()



def plot_acc_curves(training_results: dict[str, list[float]]):
    
    Train_Accuracy = training_results['train_acc']
    Test_Accuracy = training_results['val_acc']
    epochs = range(len(training_results['train_loss']))
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, Train_Accuracy, label='Train_Acc', linewidth=2)
    plt.plot(epochs, Test_Accuracy, label='Validation_Acc', linewidth=2)
    plt.title("Acc Curves", fontsize=25)
    plt.xlabel("Epochs", fontsize=20)
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.025))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.show()



def eval_model(model, dataloader, device, num_classes, is_RNN):
    test_acc = 0.0
    test_precision = [0.0] * num_classes
    test_recall = [0.0] * num_classes
    model.eval()
    with torch.no_grad():
        for X_test, y_test in dataloader:
            if is_RNN is True:
                X_test, y_test = X_test.reshape(-1, 44, 64).to(device), y_test.to(device)
            else:
                X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            y_hat = y_pred.argmax(dim=1)
            test_acc += torch.eq(y_hat, y_test).sum().item() / len(y_test)
            for i in range(num_classes):
                test_precision[i] += (sum((y_hat == i) & (y_test == i)) / sum(y_hat == i)).item()
                test_recall[i] += (sum((y_hat == i) & (y_test == i)) / sum(y_test == i)).item()
        test_acc /= len(dataloader)
        precision = (sum(test_precision) / num_classes)/ len(dataloader)
        recall = (sum(test_recall) / num_classes) / len(dataloader)
        F1_Score = 2 * precision * recall / (precision + recall)
        print("-" * 100)
        print(f"model_name: {model.__class__.__name__} | "
              f"model_acc: {test_acc:.4f} | "
              f"model_precision: {precision:.4f} | "
              f"model_recall: {recall:.4f} | "
              f"model_F1_score: {F1_Score:.4f}")
        
            