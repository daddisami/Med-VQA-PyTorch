import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

def build_vocabs(data_dir, num_ans_top=15, vocab_path=None):
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            v = json.load(f)
            return v['top_ans'], v['word2idx']

    train_json = os.path.join(data_dir, 'train_data.json')
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Fichier d'entraînement manquant pour construire le vocabulaire : {train_json}")

    with open(train_json, 'r') as f:
        train_data = json.load(f)
        
    # Vocabulaire des réponses (Top N)
    counts = {}
    for item in train_data:
        ans = str(item['answer']).lower()
        counts[ans] = counts.get(ans, 0) + 1
    top_ans = [x[0] for x in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:num_ans_top]]
    
    # Vocabulaire des questions (pour la Baseline)
    word2idx = {"<pad>": 0, "<unk>": 1}
    for item in train_data:
        tokens = word_tokenize(str(item['question']).lower())
        for w in tokens:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
                
    return top_ans, word2idx

class VQARADDataset(Dataset):
    def __init__(self, data_dir, json_file, top_ans, word2idx, img_size=(224, 224), is_vlm=False, max_q_len=20):
        self.data_dir = data_dir
        self.is_vlm = is_vlm
        self.max_q_len = max_q_len
        self.top_ans = top_ans
        self.atoi = {w: i for i, w in enumerate(top_ans)}
        self.word2idx = word2idx
        
        with open(os.path.join(data_dir, json_file), 'r') as f:
            raw_data = json.load(f)
            
        # Filtrer pour ne garder que les réponses du top_ans
        self.data = [item for item in raw_data if str(item['answer']).lower() in self.atoi]
                
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.is_vlm:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, 'images', item['image_name'])
        img = cv2.imread(img_path)
        if img is None:
            # Si l'image est corrompue ou manquante, on peut soit lever une erreur explicite 
            # soit (mieux pour l'entraînement) renvoyer un tenseur vide/placeholder ou logger l'erreur.
            # Ici, on lève une exception informative pour éviter le crash cryptique de cvtColor.
            raise FileNotFoundError(f"Impossible de lire l'image : {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)
        
        label = self.atoi[str(item['answer']).lower()]
        question = str(item['question']).lower()
        
        if self.is_vlm:
            encoded = self.tokenizer(question, padding='max_length', truncation=True, max_length=self.max_q_len, return_tensors='pt')
            return img_tensor, encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), label
        else:
            # Tokenisation standard pour la Baseline
            tokens = word_tokenize(question)
            seq = [self.word2idx.get(w, 1) for w in tokens][:self.max_q_len]
            seq += [0] * (self.max_q_len - len(seq)) # Padding
            return img_tensor, torch.tensor(seq, dtype=torch.long), label

def create_dataloaders(data_dir, batch_size=16, is_vlm=False, vocab_path=None):
    top_ans, word2idx = build_vocabs(data_dir, vocab_path=vocab_path)
    
    # Si on est en train (pas de vocab_path fourni), on crée le dossier checkpoints
    if vocab_path is None:
        os.makedirs('./checkpoints', exist_ok=True)
        with open('./checkpoints/vocab.json', 'w') as f:
            json.dump({'top_ans': top_ans, 'word2idx': word2idx}, f)

    # Séparation conforme au README
    # Pour l'évaluation pure, on pourrait ne charger que val_dataset, mais on garde la structure
    train_json = 'train_data.json'
    test_json = 'test_data.json'
    
    # On n'instancie le train_dataset que si le fichier existe (utile pour le déploiement)
    train_loader = None
    if os.path.exists(os.path.join(data_dir, train_json)):
        train_dataset = VQARADDataset(data_dir, train_json, top_ans, word2idx, is_vlm=is_vlm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = VQARADDataset(data_dir, test_json, top_ans, word2idx, is_vlm=is_vlm)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, top_ans, len(word2idx)