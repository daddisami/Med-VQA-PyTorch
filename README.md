Med-VQA-PyTorch
A Medical Visual Question Answering Framework Using ResNet50 and PubMedBERT
Med-VQA-PyTorch is a deep learning framework designed to interpret radiological imagery alongside natural language clinical queries to generate precise, medically accurate answers. By leveraging advanced vision-language multimodal fusion techniques, the system bridges the gap between complex visual diagnostics and automated clinical decision support.
Overview & Objective
The primary objective of this project is to build an efficient, multimodal diagnostic support system capable of analyzing medical visual inputs (such as X-rays, CT scans, and MRIs) alongside paired clinical questions.

The framework processes paired inputs to address key diagnostic tasks:

Visual Feature Extraction: Capturing semantic spatial patterns from radiological scans.
Clinical Language Processing: Contextualizing query semantics using domain-specific medical language models.
Multimodal Fusion: Combining visual and textual embeddings through attention mechanisms to predict accurate answers for both closed-ended (e.g., Yes/No) and open-ended clinical questions.
Dataset
This repository utilizes the VQA-RAD (Visual Question Answering in Radiology) dataset, a specialized benchmark manually constructed and validated by clinicians.

Images: Contains naturally occurring radiological images across various body regions (chest, abdomen, head) and modalities (CT, MRI, X-ray).
Question-Answer Pairs: Includes both closed-ended questions (e.g., "Is there evidence of pneumothorax?") and open-ended questions (e.g., "What abnormality is visible in the right lung upper lobe?").
Preprocessing: Images are normalized and augmented, while text sequences are tokenized using PubMedBERT's specialized clinical vocabulary.
Model Architecture
The repository provides a comparison between a baseline convolutional approach and an optimized Vision-Language Model (VLM).
1. Baseline Architecture (CNN-based)
Visual Encoder: ResNet50 pretrained backbone for standard visual feature extraction.
Text Encoder: Standard recurrent network layer for text embedding extraction.
Fusion: Simple feature concatenation followed by a multi-layer perceptron (MLP) classification head.
2. Optimized Multimodal Architecture (VLM)
Visual Encoder: ResNet50 fine-tuned to capture high-level spatial visual representation.
Textual Encoder: Hybrid setup combining LSTM for sequential flow and PubMedBERT (a domain-specific BERT pretrained on biomedical text) for deep contextual language understanding.
Fusion Mechanism: Cross-Attention Fusion Module, allowing dynamically weighted interactions between textual tokens and visual feature maps prior to final answer classification.
Experimental Results
The optimized VLM approach demonstrates significant performance gains over the standard CNN baseline across accuracy and operational latency metrics.

Metric / Question Type
Baseline (ResNet50 + Standard Text)
Optimized VLM (ResNet50 + PubMedBERT + Cross-Attention)
Overall Accuracy
58.90%
66.67%
Open-Ended Accuracy
3.80%
60.33%
Inference Latency
24.98 ms / sample
13.40 ms / sample

Key Findings
Open-Ended Generation: The optimized VLM demonstrates a dramatic increase in open-ended question accuracy (60.33% vs 3.80%), demonstrating that cross-attention paired with PubMedBERT embeddings effectively prevents the model from defaulting to naive frequent-class predictions.
Inference Efficiency: Through optimized tensor fusion and parameter-efficient cross-attention, the VLM achieves an inference runtime of 13.40 ms per sample, delivering nearly a 2x speedup compared to the baseline pipeline.
Installation & Usage
Prerequisites
Python 3.8 or higher
CUDA-compatible GPU (recommended for training and fast inference)
1. Environment Setup
Clone the repository and install the required dependencies:git clone https://github.com/your-username/Med-VQA-PyTorch.git

cd Med-VQA-PyTorch

pip install -r requirements.txt
2. Dataset Preparation
Download the VQA-RAD dataset and structure your directory as follows:data/

└── VQA_RAD/

    ├── images/

    ├── train_data.json

    └── test_data.json
3. Training the Model
Train the optimized VLM model using PubMedBERT and cross-attention fusion:h
python train.py 
--model_type vlm 
--data_dir ./data/VQA_RAD 
--batch_size 16 
--epochs 50 
--lr 2e-5

To train the baseline CNN model for comparison:

```bash

python train.py --model_type baseline --data_dir ./data/VQA_RAD
4. Evaluation & Inference
Run evaluation on the test set to compute overall accuracy, open-ended accuracy, and average latency:python evaluate.py \

  --checkpoint ./checkpoints/vlm_best.pth \

  --data_dir ./data/VQA_RAD

To run inference on a single image and clinical question pair:python infer.py \

  --image_path ./data/VQA_RAD/images/sample1.jpg \

  --question "Is there organomegaly present?" \

  --checkpoint ./checkpoints/vlm_best.pth

