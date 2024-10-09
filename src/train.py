import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
import tqdm
import time
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preprocessing and Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformations for validation and test sets
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Directory
dataset_dir = "data/road_sign_dataset"

# Load datasets from already split folders
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'val'), transform=val_test_transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Create ConvNeXt v2 Tiny model
model = create_model('convnextv2_tiny', pretrained=True, num_classes=9)
model.to(device)

# Define optimizer with learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Loss function
criterion = nn.CrossEntropyLoss()

# Function to train the model
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, early_stop_patience=5):
    best_val_loss = float('inf')
    patience = 0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Progress bar for training
        train_progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar with current metrics
            train_progress.set_postfix({
                "Train Loss": running_loss / len(train_loader),
                "Train Accuracy": 100 * correct_train / total_train
            })
        
        # Step the scheduler
        scheduler.step(running_loss / len(train_loader))

        # Validation phase
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        model.eval()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print metrics after each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            # Save the model state
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Plotting the results
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Accuracy')

    plt.show()

# Train the model
if __name__ == "__main__":
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, early_stop_patience=5)
