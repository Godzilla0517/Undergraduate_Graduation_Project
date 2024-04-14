import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from model_builder import RNN
from data_setup import DroneAudioDataset
from engine import train
from utils import split_dataset, plot_loss_curves, plot_acc_curves, save_model



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 64
SQUENCE_LENGTH = 44
NUM_LAYERS = 2
HIDDEN_SIZE = 256
NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
model_2 = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DEVICE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(model_2.parameters(), lr=LEARNING_RATE)
adjuster = torch.optim.lr_scheduler.StepLR(optimizer=trainer, step_size=7, gamma=0.1)



ANNOTATIONS_FILE = r"C:\MachineLearning\Graduation_Project\data\DroneAudio.xlsx"
AUDIO_DIR = r"C:\MachineLearning\Graduation_Project\data\DroneAudio_Mono_16K"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
TRANSFORMATION_0 = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
Drone_audio = DroneAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, TRANSFORMATION_0)
train_dataset, test_dataset = split_dataset(dataset=Drone_audio, train_radio=0.8, random_seed=None)
train_iter = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iter = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Dataset Ready!")
print("-----------------------------------------------------------------------------------------------------")



start_time = timer()
model_results = train(
    model=model_2,
    train_dataloader=train_iter,
    test_dataloader=test_iter,
    loss_fn=criterion,
    optimizer=trainer,
    scheduler=adjuster,
    epochs=EPOCHS,
    device=DEVICE,
    is_RNN=True
)
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")
save_model(model=model_2, 
           target_dir=r"C:\MachineLearning\Graduation_Project\models", 
           model_name="Model_2_RNN.pth")
plot_loss_curves(training_results=model_results)
plot_acc_curves(training_results=model_results)
