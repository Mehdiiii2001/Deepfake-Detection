import os
import gdown
import zipfile
import shutil
from tqdm import tqdm

def download_dataset():
    """
    Download the FaceForensics++ dataset (or a similar deepfake dataset)
    Note: You'll need to replace the URL with an actual dataset URL
    """
    print("Downloading dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Example URLs for different parts of the dataset
    # Note: Replace these with actual dataset URLs
    dataset_urls = {
        'real': 'YOUR_REAL_VIDEOS_URL',
        'fake': 'YOUR_FAKE_VIDEOS_URL'
    }
    
    for category, url in dataset_urls.items():
        output = f'data/{category}.zip'
        if not os.path.exists(output):
            try:
                gdown.download(url, output, quiet=False)
            except Exception as e:
                print(f"Error downloading {category} videos: {str(e)}")
                continue

def extract_and_organize():
    """Extract and organize the dataset into train/val splits"""
    print("\nExtracting and organizing dataset...")
    
    # Create directory structure
    for split in ['train', 'val']:
        for category in ['real', 'fake']:
            os.makedirs(f'data/{split}/{category}', exist_ok=True)
    
    # Extract and organize files
    for category in ['real', 'fake']:
        zip_path = f'data/{category}.zip'
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract to temporary directory
                    temp_dir = f'data/temp_{category}'
                    os.makedirs(temp_dir, exist_ok=True)
                    zip_ref.extractall(temp_dir)
                    
                    # Get list of all video files
                    videos = []
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith(('.mp4', '.avi')):
                                videos.append(os.path.join(root, file))
                    
                    # Split into train/val (80/20 split)
                    train_size = int(0.8 * len(videos))
                    train_videos = videos[:train_size]
                    val_videos = videos[train_size:]
                    
                    # Move files to appropriate directories
                    print(f"\nOrganizing {category} videos...")
                    for video in tqdm(train_videos):
                        shutil.copy2(video, f'data/train/{category}/')
                    
                    for video in tqdm(val_videos):
                        shutil.copy2(video, f'data/val/{category}/')
                    
                    # Cleanup
                    shutil.rmtree(temp_dir)
            
            except Exception as e:
                print(f"Error processing {category} videos: {str(e)}")

def verify_dataset():
    """Verify the dataset structure and count samples"""
    print("\nVerifying dataset structure...")
    
    for split in ['train', 'val']:
        for category in ['real', 'fake']:
            path = f'data/{split}/{category}'
            if os.path.exists(path):
                num_videos = len([f for f in os.listdir(path) 
                                if f.endswith(('.mp4', '.avi'))])
                print(f"{split}/{category}: {num_videos} videos")
            else:
                print(f"Warning: Directory not found - {path}")

def main():
    print("=== Deepfake Detection Dataset Preparation ===")
    
    # Download dataset
    download_dataset()
    
    # Extract and organize
    extract_and_organize()
    
    # Verify dataset
    verify_dataset()
    
    print("\nDataset preparation completed!")
    print("\nTo train the model, run:")
    print("python train_deepfake_detector.py --data_dir data")

if __name__ == '__main__':
    main() 