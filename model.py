import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class MNIST_CNN(nn.Module):
    """Простая CNN для классификации MNIST"""
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(epochs=5, batch_size=64, save_path='mnist_cnn.pth'):
    """Обучение модели на MNIST"""
    # Преобразования
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Загрузка данных
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    
    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNIST_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    train_losses = []
    train_accuracies = []
    
    print(f"Training on {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Training Accuracy: {accuracy:.2f}%\n')
    
    # Тестирование
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)')
    
    # Сохранение модели
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    
    return model, train_losses, train_accuracies, test_accuracy

def load_model(model_path='mnist_cnn.pth'):
    """Загрузка предобученной модели"""
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

if __name__ == "__main__":
    # Обучить модель, если скрипт запущен напрямую
    train_model()