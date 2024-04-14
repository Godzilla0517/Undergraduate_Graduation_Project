import os
import pandas as pd
import torchaudio
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils import  MonoToColor



class DroneAudioDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, target_sr, num_samples, transformation):
        self.annotations = pd.read_excel(annotations_file)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.num_samples = num_samples
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        sample_path = self._get_sample_path(index)
        label = self._get_sample_label(index)
        signal, sr = torchaudio.load(sample_path)
        signal = self._resample(signal, sr)
        signal = self._cut(signal)
        signal = self._pad(signal)
        signal = self.transformation(signal)
        return signal, label

    def _get_sample_path(self, index):
        folder = f"{self.annotations.iloc[index, 1]}"
        path = os.path.join(self.audio_dir, folder, self.annotations.iloc[index, 0])
        return path
    
    def _get_sample_label(self, index):
        return self.annotations.iloc[index, 2]
    
    def _resample(self, signal, sr):
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)    
        return signal
    
    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing = self.num_samples - length_signal
            last_dim_padding = (0, num_missing)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    


if __name__ == "__main__":

    ANNOTATIONS_FILE = r"C:\MachineLearning\Graduation_Project\data\DroneAudio.xlsx"
    AUDIO_DIR = r"C:\MachineLearning\Graduation_Project\data\DroneAudio_Mono_16K"
    SAMPLE_RATE = 22050 
    NUM_SAMPLES = 22050
    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=64
    )
    
    drone_audio = DroneAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, MEL_SPECTROGRAM)
    print(f"There are {len(drone_audio)} samples in the dataset.")
    signal, label = drone_audio[1]
    print(signal)
    print(signal.shape)