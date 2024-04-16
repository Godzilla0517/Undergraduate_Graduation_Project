import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from model_builder import CustomCNN
from data_setup import DroneAudioDataset
from utils import split_dataset, plot_loss_curves, plot_acc_curves, eval_model
from engine import train
from torchvision import transforms



BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANNOTATIONS_FILE = r"C:\MachineLearning\Graduation_Project\data\DroneAudio.xlsx"
AUDIO_DIR = r"C:\MachineLearning\Graduation_Project\data\DroneAudio_Mono_16K"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
TRANSFORMATION_0 = transforms.Compose([
    MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64), 
    AmplitudeToDB(stype="power", top_db=80)
])



model_0 = CustomCNN(num_classes=6).to(DEVICE)
criterion = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(model_0.parameters(), lr=LEARNING_RATE)
Drone_audio = DroneAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, TRANSFORMATION_0)
adjuster = torch.optim.lr_scheduler.StepLR(optimizer=trainer, step_size=7, gamma=0.1)



Drone_audio = DroneAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, TRANSFORMATION_0)
train_dataset, val_dataset, test_dataset = split_dataset(
    dataset=Drone_audio, 
    train_radio=0.8, 
    test_ratio=0.1, 
    random_seed=42)
train_iter = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_iter = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_iter = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
print("-" * 100)
print("Dataset Ready!")
print("-" * 100)


start_time = timer()
model_results = train(
    model=model_0,
    train_dataloader=train_iter,
    val_dataloader=val_iter,
    loss_fn=criterion,
    optimizer=trainer,
    scheduler=adjuster,
    epochs=EPOCHS,
    device=DEVICE,
    is_RNN=False
)
end_time = timer()
total_time_model_0 = end_time - start_time
print("-" * 100)
print(f"Total training time: {total_time_model_0:.3f} seconds")
plot_loss_curves(training_results=model_results)
plot_acc_curves(training_results=model_results)



torch.save(
    obj=model_0.state_dict(), 
    f=r"C:\MachineLearning\Graduation_Project\models\model_0_CustomCNN.pth"
)
loaded_model_0 = CustomCNN(num_classes=6)
loaded_model_0.load_state_dict(
    torch.load(f=r"C:\MachineLearning\Graduation_Project\models\model_0_CustomCNN.pth"))
loaded_model_0 = loaded_model_0.to(DEVICE)



eval_model(
    model=loaded_model_0, 
    dataloader=test_iter,
    device=DEVICE,
    num_classes=6,
    is_RNN=False
)
print("-" * 100)
