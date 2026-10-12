import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import create_dataloaders
from models import CNNBaseline, VLMOptimized
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('./checkpoints', exist_ok=True)
    
    is_vlm = args.model_type == 'vlm'
    train_loader, val_loader, top_ans, vocab_size = create_dataloaders(args.data_dir, batch_size=args.batch_size, is_vlm=is_vlm)
    
    if is_vlm:
        model = VLMOptimized(num_classes=len(top_ans)).to(device)
    else:
        model = CNNBaseline(vocab_size=vocab_size, num_classes=len(top_ans)).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            
            if is_vlm:
                images, input_ids, attention_mask, labels = batch
                images, input_ids, attention_mask, labels = images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(images, input_ids, attention_mask)
            else:
                images, text_seqs, labels = batch
                images, text_seqs, labels = images.to(device), text_seqs.to(device), labels.to(device)
                outputs = model(images, text_seqs)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Validation]"):
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
                
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"./checkpoints/{args.model_type}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Med-VQA-PyTorch models")
    parser.add_argument('--model_type', type=str, choices=['vlm', 'baseline'], required=True, help="Model type to train")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    train(args)