import torch
import torch.nn as nn
from torchvision.models import resnet18


class CustomCNN(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )        
        self.layer_4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(128 * 5 * 4, num_classes)            
        )
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        predictions = self.classifier(x)
        return predictions


class ResNet18(nn.Module):
    
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet_18 = resnet18(pretrained=True)
        self.resnet_18.fc = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.resnet_18(x)
        return x


class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        
    def forward(self, x):
        hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.rnn(x, hidden_0)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class GRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        
    def forward(self, x):
        hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.gru(x, hidden_0)
        x = x[:, -1, :]
        x = self.fc(x)
        return x        
    
    
class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        
    def forward(self, x):
        hidden_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        cell_state_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (hidden_0, cell_state_0))
        x = x[:, -1, :]
        x = self.fc(x)
        return x 
    
    
if __name__ == "__main__":
    a = torch.randn(32, 3, 64, 44)
    model = ResNet18(num_classes=6)
    y = model(a)
    print(y)
    print(y.shape)