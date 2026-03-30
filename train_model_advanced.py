import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Dataset Class (Unchanged)
class FER2013Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# --- Load and Prepare Data ---
print("Loading data...")
data = pd.read_csv('fer2013.csv')
pixels = data['pixels'].tolist()
faces = [np.asarray([int(p) for p in p_seq.split(' ')]).reshape(48, 48).astype('uint8') for p_seq in pixels]
emotions = np.argmax(pd.get_dummies(data['emotion']).to_numpy(), axis=1)

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42, stratify=emotions)

# --- Calculate Class Weights ---
print("Calculating class weights for imbalanced dataset...")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# --- THIS IS THE CORRECT, COMPLETE TRANSFORMS SECTION ---
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channels
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- DataLoaders ---
train_dataset = FER2013Dataset(X_train, y_train, transform=data_transforms['train'])
test_dataset = FER2013Dataset(X_test, y_test, transform=data_transforms['val'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Load Model and Prepare for Fine-Tuning ---
print("Loading pre-trained ResNet50 and preparing for fine-tuning...")
model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')

# Unfreeze the final block (layer4) and the classifier
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 7))

# --- Prepare for Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- The Advanced Training Loop ---
print("Starting advanced training...")
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# --- Save the Final Model ---
print("Training finished. Saving advanced model...")
torch.save(model.state_dict(), 'my_emotion_model_advanced.pth')
print("Model saved as my_emotion_model_advanced.pth")