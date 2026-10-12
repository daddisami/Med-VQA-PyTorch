import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input

# Ensure nltk punkt is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_and_preprocess_data(json_path, image_path, img_size, num_ans_top, vlm_mode=False, max_q_len=None):
    """
    Loads and preprocesses the VQA-RAD dataset.
    
    If vlm_mode is True, it also prepares text data (tokenization, padding).
    """
    print("--- Loading Data ---")
    with open(json_path, 'r') as f: 
        raw_data = json.load(f)

    # Top frequent labels logic
    counts = {}
    for item in raw_data:
        ans = str(item['answer']).lower()
        counts[ans] = counts.get(ans, 0) + 1
    top_ans = [x[0] for x in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:num_ans_top]]
    atoi = {w: i for i, w in enumerate(top_ans)}

    # Load Data & Metadata
    X_images, Y_ids, questions_text = [], [], []
    filenames = []

    print("Preparing Images & Texts...")
    for item in tqdm(raw_data):
        ans = str(item['answer']).lower()
        if ans in atoi:
            path = os.path.join(image_path, item['image_name'])
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_images.append(img)
                Y_ids.append(atoi[ans])
                questions_text.append(str(item['question']).lower())
                filenames.append(item['image_name'])

    X_images = preprocess_input(np.array(X_images))
    Y_ids = np.array(Y_ids)
    Y_one_hot = to_categorical(Y_ids, num_classes=num_ans_top)
    filenames = np.array(filenames)
    
    if not vlm_mode:
        questions_text = np.array(questions_text)
        return X_images, Y_ids, Y_one_hot, filenames, questions_text, top_ans

    # --- NLP PREP for VNM ---
    print("Tokenization...")
    tokens = [word_tokenize(q) for q in questions_text]
    word_index = {}
    idx = 1
    for seq in tokens:
        for w in seq:
            if w not in word_index: 
                word_index[w] = idx
                idx += 1
                
    X_text = pad_sequences([[word_index[w] for w in seq] for seq in tokens], maxlen=max_q_len, padding='post')
    vocab_size = len(word_index) + 1
    
    return X_images, Y_ids, Y_one_hot, filenames, questions_text, top_ans, X_text, word_index, vocab_size

