import argparse
import torch
import json
from data_loader import create_dataloaders
from models import CNNBaseline, VLMOptimized
from tqdm import tqdm

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    is_vlm = args.model_type == 'vlm'
    
    # On utilise le vocabulaire sauvegardé s'il existe pour éviter de dépendre de train_data.json
    vocab_path = './checkpoints/vocab.json'
    
    _, val_loader, top_ans, vocab_size = create_dataloaders(
        args.data_dir, 
        batch_size=16, 
        is_vlm=is_vlm, 
        vocab_path=vocab_path if os.path.exists(vocab_path) else None
    )
    
    if is_vlm:
        model = VLMOptimized(num_classes=len(top_ans))
    else:
        model = CNNBaseline(vocab_size=vocab_size, num_classes=len(top_ans))
        
    # Load weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if is_vlm:
                images, input_ids, attention_mask, labels = batch
                images, input_ids, attention_mask, labels = images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(images, input_ids, attention_mask)
            else:
                images, text_seqs, labels = batch
                images, text_seqs, labels = images.to(device), text_seqs.to(device), labels.to(device)
                outputs = model(images, text_seqs)
                
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