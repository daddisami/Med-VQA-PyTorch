import argparse
import torch
import cv2
from torchvision import transforms
from transformers import AutoTokenizer
from models import VLMOptimized

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    # Note: Dans un vrai scénario, il faudrait sauvegarder et recharger la liste des 'top_ans' (vocabulaire des réponses)
    # Ici on simule avec une liste factice pour l'exemple.
    top_ans = ["yes", "no", "normal", "lung", "chest", "right", "left", "pneumonia", "pleural effusion", "cardiomegaly", "mass", "nodule", "opacity", "infiltrate", "consolidation"]
    
    model = VLMOptimized(num_classes=len(top_ans))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Process Image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not read image at {args.image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
    
    # Process Text
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    encoded = tokenizer(args.question, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor, input_ids, attention_mask)
        _, predicted_idx = torch.max(output.data, 1)
        
    answer = top_ans[predicted_idx.item()]
    
    print("-" * 30)
    print(f"Question: {args.question}")
    print(f"Predicted Answer: {answer}")
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with Med-VQA VLM")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--question', type=str, required=True, help="Clinical question")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to VLM model weights (.pth)")
    
    args = parser.parse_args()
    infer(args)
