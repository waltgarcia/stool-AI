#!/usr/bin/env python3
"""
Retrain Model - Train an improved model using user submissions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    data_dir = "data/bristol_stool_dataset"
    img_size = 224
    batch_size = 16
    epochs = 30
    learning_rate = 0.0001
    num_classes = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = "model_weights.pth"
    checkpoint_dir = "checkpoints"

# Custom Dataset Class
class BristolDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        for class_idx in range(1, Config.num_classes + 1):
            class_dir = os.path.join(data_dir, f"type_{class_idx}")
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx - 1)  # 0-based indexing
        
        if len(self.images) == 0:
            print(f"⚠️ No images found in {data_dir}")
            raise RuntimeError("Dataset is empty")
        
        # Split dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.images, self.labels, test_size=0.3, random_state=42, stratify=self.labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        if split == 'train':
            self.images, self.labels = X_train, y_train
        elif split == 'val':
            self.images, self.labels = X_val, y_val
        else:  # test
            self.images, self.labels = X_test, y_test
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image and label
            blank = Image.new('RGB', (224, 224))
            if self.transform:
                blank = self.transform(blank)
            return blank, label

# Model Architecture (same as in app)
class BristolClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(BristolClassifier, self).__init__()
        try:
            self.backbone = models.resnet50(weights=None)
        except Exception:
            self.backbone = models.resnet50(pretrained=False)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({'loss': running_loss/(total/Config.batch_size), 
                                  'acc': 100.*correct/total})
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({'loss': val_loss/(total/Config.batch_size), 
                                    'acc': 100.*correct/total})
        
        val_loss = val_loss/len(val_loader)
        val_acc = 100.*correct/total
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"✅ Saved best model with validation accuracy: {val_acc:.2f}%")
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
    
    return train_losses, val_losses, train_accs, val_accs

# Data augmentation
train_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    print(f"🔧 Using device: {Config.device}")
    print(f"📁 Data directory: {Config.data_dir}")
    
    # Create datasets
    try:
        train_dataset = BristolDataset(Config.data_dir, split='train', transform=train_transform)
        val_dataset = BristolDataset(Config.data_dir, split='val', transform=val_transform)
        test_dataset = BristolDataset(Config.data_dir, split='test', transform=val_transform)
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have images in:")
        for i in range(1, 8):
            print(f"   - {Config.data_dir}/type_{i}/")
        return
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, 
                            shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, 
                          shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = BristolClassifier(num_classes=Config.num_classes)
    model = model.to(Config.device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3, 
                                                     verbose=True)
    
    # Train model
    print("\n🚀 Starting training...\n")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, Config.epochs
    )
    
    print("\n✅ Training completed!")
    print(f"📊 Best validation accuracy: {max(val_accs):.2f}%")

if __name__ == "__main__":
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    main()
