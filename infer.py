import argparse
import torch
import cv2
import json
from torchvision import transforms
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from models import VLMOptimized, CNNBaseline

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Chargement dynamique du vocabulaire
    with open('./checkpoints/vocab.json', 'r') as f:
        vocabs = json.load(f)
    top_ans = vocabs['top_ans']
    word2idx = vocabs['word2idx']
    
    is_vlm = args.model_type == 'vlm'
    
    if is_vlm:
        model = VLMOptimized(num_classes=len(top_ans))
    else:
        model = CNNBaseline(vocab_size=len(word2idx), num_classes=len(top_ans))
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    question = args.question.lower()
    
    with torch.no_grad():
        if is_vlm:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            encoded = tokenizer(question, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            output = model(img_tensor, input_ids, attention_mask)
        else:
            tokens = word_tokenize(question)
            seq = [word2idx.get(w, 1) for w in tokens][:20]
            seq += [0] * (20 - len(seq))
            text_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            output = model(img_tensor, text_tensor)
            
        _, predicted_idx = torch.max(output.data, 1)
        
    print(f"Question: {args.question}")
    print(f"Predicted Answer: {top_ans[predicted_idx.item()]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['vlm', 'baseline'], required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    infer(args)