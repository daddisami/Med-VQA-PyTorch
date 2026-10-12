import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

class CNNBaseline(nn.Module):
    def __init__(self, num_classes=15):
        super(CNNBaseline, self).__init__()
        # Image Encoder
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Retirer la couche fc finale
        self.resnet.fc = nn.Identity()
        
        # Text Encoder (LSTM basique)
        self.embedding = nn.Embedding(10000, 300) # Vocabulaire simplifié pour la baseline
        self.lstm = nn.LSTM(300, 256, batch_first=True)
        
        # MLP de classification
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_seqs=None):
        # Features visuelles
        img_features = self.resnet(images) # [B, 2048]
        
        if text_seqs is not None:
            # Features textuelles (simulation si on passait des séquences à la baseline)
            embedded = self.embedding(text_seqs)
            _, (hidden, _) = self.lstm(embedded)
            text_features = hidden[-1] # [B, 256]
            
            # Concaténation
            combined = torch.cat((img_features, text_features), dim=1)
        else:
            # Simplification si on ne passe que des images pour tester
            text_features = torch.zeros(img_features.size(0), 256).to(img_features.device)
            combined = torch.cat((img_features, text_features), dim=1)
            
        out = self.classifier(combined)
        return out

class VLMOptimized(nn.Module):
    def __init__(self, num_classes=15):
        super(VLMOptimized, self).__init__()
        # 1. Visual Encoder: ResNet50 (on garde les feature maps spatiales)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.visual_extractor = nn.Sequential(*list(resnet.children())[:-2]) # [B, 2048, 7, 7]
        
        # Projection pour réduire la dimension visuelle
        self.vis_proj = nn.Conv2d(2048, 768, kernel_size=1) 
        
        # 2. Textual Encoder: PubMedBERT
        self.text_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
        # 3. Cross-Attention Fusion Module
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Extraire les features visuelles
        v_features = self.visual_extractor(images) # [B, 2048, 7, 7]
        v_features = self.vis_proj(v_features) # [B, 768, 7, 7]
        
        # Aplatir les dimensions spatiales [B, 768, 49] puis transposer [B, 49, 768] (Keys & Values)
        v_features_flat = v_features.view(v_features.size(0), v_features.size(1), -1).transpose(1, 2)
        
        # Extraire les features textuelles avec PubMedBERT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_features = text_outputs.last_hidden_state # [B, seq_len, 768] (Queries)
        
        # Cross-Attention: Texte (Query) assiste Image (Key/Value)
        attn_output, _ = self.cross_attention(query=t_features, key=v_features_flat, value=v_features_flat)
        
        # Pooling (utiliser le token [CLS] qui est à l'index 0)
        pooled_output = attn_output[:, 0, :] # [B, 768]
        
        # Classification
        out = self.classifier(pooled_output)
        return out
