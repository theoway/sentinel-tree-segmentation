import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
from scipy.ndimage import zoom
from tqdm import tqdm
import os
import re
import csv

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, image_files, labels_folder, target_size = (512, 512), transform=None):
        self.image_files = image_files
        self.labels_folder = labels_folder
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = self.image_files[idx]
        match = re.search(r'(\d+)', image_path)        
        data_id = match.group(1)
        label_path = os.path.join(self.labels_folder, data_id + '.npy')

        image = np.load(image_path)
        label = np.load(label_path)
        image = image.astype(np.float32)
        mask = np.where(label >= 35, 1, 0)
        mask = np.reshape(mask, (mask.shape[1], mask.shape[2], 1))
        
        image = self.interpolate(image, self.target_size)
        mask = self.interpolate(mask, self.target_size)
        image[..., 0 : 3] /= 255
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    
    def interpolate(self, array, target_size):
        zoom_factors = [target_size[i] / array.shape[i] for i in range(2)]
        zoom_factors.append(1)
        
        interpolated_array = zoom(array, zoom_factors, order=1, mode='nearest')

        return interpolated_array
    
images_folder = "../data/sat_imgs/"
labels_folder = "../data/labels/"

image_files = os.listdir(images_folder)
image_files = [os.path.join(images_folder, file) for file in image_files]
train_files, test_files = train_test_split(image_files, test_size=0.3, random_state=42)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CustomDataset(train_files, labels_folder, transform=transform)
test_dataset = CustomDataset(test_files, labels_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

in_channels = 13  
out_channels = 1  
model = UNet(in_channels, out_channels)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)


csv_file_path = 'training_metrics.csv'
fieldnames = ['Epoch', 'Train_Loss', 'Train_Accuracy', 'Val_Loss', 'Val_Accuracy']
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

print("Training...")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    train_predictions = []
    train_labels = []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        labels = labels.float()

        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
        train_predictions.extend(predicted_labels.cpu().numpy().flatten())
        train_labels.extend(labels.cpu().numpy().flatten())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_accuracy = accuracy_score(train_labels, train_predictions)

    model.eval()
    total_val_loss = 0
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            labels = labels.float()

            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            val_predictions.extend(predicted_labels.cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())

    val_accuracy = accuracy_score(val_labels, val_predictions)

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {total_train_loss / len(train_loader)} - Train Accuracy: {train_accuracy}")
    print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {total_val_loss / len(test_loader)} - Val Accuracy: {val_accuracy}")

    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writerow({'Epoch': epoch + 1, 'Train_Loss': total_train_loss / len(train_loader),
                             'Train_Accuracy': train_accuracy, 'Val_Loss': total_val_loss / len(test_loader),
                             'Val_Accuracy': val_accuracy})
        
torch.save(model.state_dict(), 'unet_model.pth')
print("Model saved in the current directory")