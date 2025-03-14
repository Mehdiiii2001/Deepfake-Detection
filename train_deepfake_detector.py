import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from deepfake_detector import DeepfakeDetector, train_step, evaluate
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train Deepfake Detector')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=16,
                      help='Number of frames per video clip')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                      choices=['efficientnet_b0', 'resnet50'],
                      help='CNN backbone architecture')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--no_adversarial', action='store_true',
                      help='Disable adversarial training')
    return parser.parse_args()

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_frames=16, transform=None, train=True):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.transform = transform
        self.train = train
        
        # Split data into real and fake directories
        self.real_dir = os.path.join(data_dir, 'train' if train else 'val', 'real')
        self.fake_dir = os.path.join(data_dir, 'train' if train else 'val', 'fake')
        
        # Get video paths
        self.real_videos = [os.path.join(self.real_dir, f) for f in os.listdir(self.real_dir)]
        self.fake_videos = [os.path.join(self.fake_dir, f) for f in os.listdir(self.fake_dir)]
        
        # Combine and create labels
        self.videos = self.real_videos + self.fake_videos
        self.labels = [0] * len(self.real_videos) + [1] * len(self.fake_videos)
    
    def __len__(self):
        return len(self.videos)
    
    def load_video(self, video_path):
        """Load video frames and apply preprocessing"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while frame_count < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # If we don't have enough frames, loop the video
        while len(frames) < self.num_frames:
            frames.extend(frames[:self.num_frames - len(frames)])
        
        # Stack frames
        frames = torch.stack(frames)
        return frames
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        # Load and preprocess video frames
        frames = self.load_video(video_path)
        
        return frames, torch.tensor(label, dtype=torch.float)

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = VideoDataset(args.data_dir, args.num_frames, transform, train=True)
    val_dataset = VideoDataset(args.data_dir, args.num_frames, transform, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = DeepfakeDetector(num_frames=args.num_frames, backbone=args.backbone)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.5, patience=5,
                                                    verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_acc = train_step(
            model, optimizer, criterion, train_loader, device,
            adversarial=not args.no_adversarial
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            torch.save(checkpoint,
                      os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print("Saved best model checkpoint")
        
        # Save latest model
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        torch.save(checkpoint,
                  os.path.join(args.checkpoint_dir, 'latest_model.pth'))

if __name__ == '__main__':
    main() 