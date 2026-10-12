import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

class CNNBaseline(nn.Module):
    def __init__(self, vocab_size, num_classes=15):
        super(CNNBaseline, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        
        # Le padding index est 0 d'après notre data_loader
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0) 
        self.lstm = nn.LSTM(300, 256, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_seqs):
        img_features = self.resnet(images) # [B, 2048]
        
        # Extraction textuelle effective
        embedded = self.embedding(text_seqs)
        _, (hidden, _) = self.lstm(embedded)
        text_features = hidden[-1] # [B, 256]
        
        combined = torch.cat((img_features, text_features), dim=1)
        return self.classifier(combined)

class VLMOptimized(nn.Module):
    def __init__(self, num_classes=15):
        super(VLMOptimized, self).__init__()
        # 1. Visual Encoder: ResNet50 (on garde les feature maps spatiales)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.visual_extractor = nn.Sequential(*list(resnet.children())[:-2]) # Output: [B, 2048, 7, 7]
        
        # Projection pour aligner la dimension visuelle avec le texte (768)
        self.vis_proj = nn.Conv2d(2048, 768, kernel_size=1)
        
        # 2. Textual Encoder: Hybride PubMedBERT + LSTM
        self.bert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        # L'unité LSTM pour capturer le "sequential flow" sur les embeddings BERT
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        
        # 3. Cross-Attention Fusion Module
        # Query: Texte (LSTM outputs), Key/Value: Image (Feature Maps)
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Features visuelles
        v_features = self.visual_extractor(images) # [B, 2048, 7, 7]
        v_features = self.vis_proj(v_features) # [B, 768, 7, 7]
        # Aplatir spatialement: [B, 768, 49] -> [B, 49, 768] (pour MultiheadAttention)
        v_features_flat = v_features.view(v_features.size(0), v_features.size(1), -1).transpose(1, 2)
        
        # Features textuelles
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t_embeddings = bert_outputs.last_hidden_state # [B, seq_len, 768]
        t_features, _ = self.lstm(t_embeddings) # [B, seq_len, 768]
        
        # Cross-Attention: Le texte "interroge" l'image
        attn_output, _ = self.cross_attention(query=t_features, key=v_features_flat, value=v_features_flat)
        
        # Pooling: On utilise le premier token (souvent porteur de la sémantique globale après attention)
        pooled_output = attn_output[:, 0, :] # [B, 768]
        
        return self.classifier(pooled_output)