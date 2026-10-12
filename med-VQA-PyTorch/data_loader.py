import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import numpy as np

class VQARADDataset(Dataset):
    def __init__(self, data_dir, json_file, img_size=(224, 224), num_ans_top=15, is_vlm=False, max_q_len=20):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_vlm = is_vlm
        self.max_q_len = max_q_len
        
        json_path = os.path.join(data_dir, json_file)
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        # Obtenir les réponses les plus fréquentes
        counts = {}
        for item in raw_data:
            ans = str(item['answer']).lower()
            counts[ans] = counts.get(ans, 0) + 1
        self.top_ans = [x[0] for x in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:num_ans_top]]
        self.atoi = {w: i for i, w in enumerate(self.top_ans)}
        
        # Filtrer les données
        self.data = []
        for item in raw_data:
            ans = str(item['answer']).lower()
            if ans in self.atoi:
                self.data.append(item)
                
        # Transformations d'images (PyTorch standard pour ResNet)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Tokenizer Hugging Face pour VLM
        if self.is_vlm:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Charger l'image
        img_path = os.path.join(self.data_dir, 'images', item['image_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)
        
        # Obtenir le label
        ans = str(item['answer']).lower()
        label = self.atoi[ans]
        
        question = str(item['question']).lower()
        
        if self.is_vlm:
            # Tokenisation PubMedBERT
            encoded = self.tokenizer(question, padding='max_length', truncation=True, max_length=self.max_q_len, return_tensors='pt')
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            return img_tensor, input_ids, attention_mask, label, item['image_name'], question
        else:
            return img_tensor, question, label, item['image_name']

def create_dataloaders(data_dir, batch_size=16, is_vlm=False):
    # Pour simplifier, on utilise le même fichier pour train et val dans cet exemple, 
    # mais en pratique vous devriez utiliser train_data.json et test_data.json
    dataset = VQARADDataset(data_dir, 'VQA_RAD_Dataset_Public.json', is_vlm=is_vlm)
    
    # Split train/val (simplifié)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.top_ans
