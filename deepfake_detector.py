import torch
import torch.nn as nn
import torchvision.models as models
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True):
        super(SpatialFeatureExtractor, self).__init__()
        # Load pretrained CNN backbone
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        
        # Remove the classification head
        if backbone == 'efficientnet_b0':
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'resnet50':
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply spatial attention
        attention = self.spatial_attention(features)
        attended_features = features * attention
        
        return attended_features

class TemporalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        
        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Apply temporal attention
        attention = self.temporal_attention(encoded)
        attended_features = encoded * attention
        
        return attended_features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DeepfakeDetector(nn.Module):
    def __init__(self, num_frames=16, backbone='efficientnet_b0'):
        super(DeepfakeDetector, self).__init__()
        
        # Spatial feature extractor (CNN)
        self.spatial_extractor = SpatialFeatureExtractor(backbone=backbone)
        
        # Feature dimension reduction
        self.feature_reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.spatial_extractor.feature_dim, 512)
        )
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=512,
            nhead=8,
            num_layers=6
        )
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize adversarial training parameters
        self.register_buffer('epsilon', torch.tensor(0.3))
        self.register_buffer('alpha', torch.tensor(0.01))
        self.register_buffer('num_steps', torch.tensor(10))
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        
        # Reshape input for spatial feature extraction
        x = x.view(-1, c, h, w)
        
        # Extract spatial features
        spatial_features = self.spatial_extractor(x)
        
        # Reduce feature dimensions
        features = self.feature_reduction(spatial_features)
        
        # Reshape for temporal analysis
        features = features.view(batch_size, num_frames, -1)
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(features)
        
        # Global temporal pooling
        global_features = torch.mean(temporal_features, dim=1)
        
        # Final classification
        output = self.fusion(global_features)
        
        return output
    
    def adversarial_training_step(self, x, y, criterion):
        """
        Perform adversarial training step using PGD attack
        """
        # Store original parameters
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        
        for _ in range(self.num_steps):
            # Forward pass
            output = self(x_adv)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial example
            grad = x_adv.grad.data
            x_adv = x_adv + self.alpha * grad.sign()
            
            # Project back to epsilon ball
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            x_adv.requires_grad = True
        
        return x_adv

def train_step(model, optimizer, criterion, data_loader, device, adversarial=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (frames, targets) in enumerate(data_loader):
        frames, targets = frames.to(device), targets.to(device)
        
        # Adversarial training
        if adversarial:
            frames = model.adversarial_training_step(frames, targets, criterion)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy

def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, targets in data_loader:
            frames, targets = frames.to(device), targets.to(device)
            outputs = model(frames)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy 