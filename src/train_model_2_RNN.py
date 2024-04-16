import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from model_builder import RNN
from data_setup import DroneAudioDataset
from engine import train
from utils import split_dataset, plot_loss_curves, plot_acc_curves, eval_model
from torchvision import transforms



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 64
SQUENCE_LENGTH = 44
NUM_LAYERS = 2
HIDDEN_SIZE = 256
NUM_CLASSES = 6
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
TRANSFORMATION_0 = transforms.Compose([
    MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64), 
    AmplitudeToDB(stype="power", top_db=80)
])



Drone_audio = DroneAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, TRANSFORMATION_0)
train_dataset, val_dataset, test_dataset = split_dataset(Drone_audio, train_radio=0.8, test_ratio=0.1, random_seed=42)
train_iter = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_iter = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iter = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
print("-" * 100)
print("Dataset Ready!")
print("-" * 100)


start_time = timer()
model_results = train(
    model=model_2,
    train_dataloader=train_iter,
    val_dataloader=val_iter,
    loss_fn=criterion,
    optimizer=trainer,
    scheduler=adjuster,
    epochs=EPOCHS,
    device=DEVICE,
    is_RNN=True
)
end_time = timer()
total_time_model_2 = end_time - start_time
print("-" * 100)
print(f"Total training time: {total_time_model_2:.3f} seconds")
plot_loss_curves(training_results=model_results)
plot_acc_curves(training_results=model_results)


torch.save(
    obj=model_2.state_dict(), 
    f=r"C:\MachineLearning\Graduation_Project\models\model_2_RNN.pth"
)
loaded_model_2 = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DEVICE)
loaded_model_2.load_state_dict(
    torch.load(f=r"C:\MachineLearning\Graduation_Project\models\model_2_RNN.pth"))
loaded_model_2 = loaded_model_2.to(DEVICE)

eval_model(
    model=loaded_model_2,
    dataloader=test_iter,
    device=DEVICE,
    num_classes=6, 
    is_RNN=True
)
print("-" * 100)