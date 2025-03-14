import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import argparse
from deepfake_detector import DeepfakeDetector
from PIL import Image
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Deepfake Detection Prediction')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                      help='Path to trained model checkpoint')
    parser.add_argument('--num_frames', type=int, default=16,
                      help='Number of frames to analyze')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                      choices=['efficientnet_b0', 'resnet50'],
                      help='CNN backbone architecture')
    return parser.parse_args()

def load_video(video_path, num_frames, transform):
    """Load and preprocess video frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Get total frames and calculate sampling interval
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)
    
    frame_positions = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    for pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            if transform:
                frame = transform(frame)
            frames.append(frame)
        else:
            break
    
    cap.release()
    
    # If we don't have enough frames, duplicate the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    # Stack frames
    frames = torch.stack(frames)
    return frames

def predict_video(model, video_path, num_frames, transform, device):
    """Make prediction on a video"""
    # Load and preprocess video
    frames = load_video(video_path, num_frames, transform)
    frames = frames.unsqueeze(0)  # Add batch dimension
    frames = frames.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(frames)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize model
    model = DeepfakeDetector(num_frames=args.num_frames, backbone=args.backbone)
    
    # Load trained weights
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Make prediction
    prediction, probability = predict_video(model, args.input, args.num_frames, transform, device)
    
    # Print results
    result = "FAKE" if prediction == 1 else "REAL"
    print(f"\nPrediction Results for: {args.input}")
    print(f"Classification: {result}")
    print(f"Confidence: {probability:.2%}")
    
    # Optional: Create visualization
    cap = cv2.VideoCapture(args.input)
    ret, frame = cap.read()
    if ret:
        # Add prediction text to frame
        text = f"{result} ({probability:.2%})"
        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Save result
        output_path = f"result_{os.path.basename(args.input)}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"\nVisualization saved as: {output_path}")
    
    cap.release()

if __name__ == '__main__':
    main() 