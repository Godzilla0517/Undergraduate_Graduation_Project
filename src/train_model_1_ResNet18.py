import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from model_builder import CustomCNN, ModifiedResnet18
from data_setup import DroneAudioDataset
from utils import split_dataset, plot_loss_curves, plot_acc_curves, MonoToColor, save_model
from engine_CNN import train
from torchvision import transforms



BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANNOTATIONS_FILE = r"C:\MachineLearning\Graduation_Project\data\DroneAudio.xlsx"
AUDIO_DIR = r"C:\MachineLearning\Graduation_Project\data\DroneAudio_Mono_16K"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
TRANSFORMATION_0 = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
TRANSFORMATION_1 = transforms.Compose([TRANSFORMATION_0, AmplitudeToDB(stype="power", top_db=80), MonoToColor()])



model_0 = CustomCNN().to(DEVICE)
model_1 = ModifiedResnet18().to(DEVICE)
criterion = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(model_1.parameters(), lr=LEARNING_RATE)
Drone_audio = DroneAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, TRANSFORMATION_1)
adjuster = torch.optim.lr_scheduler.StepLR(optimizer=trainer, step_size=7, gamma=0.1)



train_dataset, test_dataset = split_dataset(dataset=Drone_audio, train_radio=0.8, random_seed=None)
train_iter = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iter = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Dataset Ready!")



start_time = timer()
model_results = train(
    model=model_1,
    train_dataloader=train_iter,
    test_dataloader=test_iter,
    loss_fn=criterion,
    optimizer=trainer,
    scheduler=adjuster,
    epochs=EPOCHS,
    device=DEVICE
)
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")
save_model(model=model_1, 
           target_dir=r"C:\MachineLearning\Graduation_Project\models", 
           model_name="Model_1_ModifiedResNet18.pth")
plot_loss_curves(training_results=model_results)
plot_acc_curves(training_results=model_results)