import argparse
import torch
from data_loader import create_dataloaders
from models import CNNBaseline, VLMOptimized
from tqdm import tqdm

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    is_vlm = args.model_type == 'vlm'
    _, val_loader, top_ans = create_dataloaders(args.data_dir, batch_size=16, is_vlm=is_vlm)
    
    if is_vlm:
        model = VLMOptimized(num_classes=len(top_ans))
    else:
        model = CNNBaseline(num_classes=len(top_ans))
        
    # Load weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if is_vlm:
                images, input_ids, attention_mask, labels, _, _ = batch
                images, input_ids, attention_mask, labels = images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(images, input_ids, attention_mask)
            else:
                images, _, labels, _ = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Overall Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Med-VQA-PyTorch models")
    parser.add_argument('--model_type', type=str, choices=['vlm', 'baseline'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    
    args = parser.parse_args()
    evaluate(args)
