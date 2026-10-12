import os
import numpy as np
import pickle
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

from config import *
from data_loader import load_and_preprocess_data
from models import build_cnn_baseline, build_vlm_model
import argparse

def train(model_type):
    VNM_MODE = model_type == 'vlm'
    
    if VNM_MODE:
        X_images, Y_ids, Y_one_hot, filenames, questions_text, top_ans, X_text, word_index, vocab_size = \
            load_and_preprocess_data(JSON_PATH, IMAGE_PATH, IMG_SIZE, NUM_ANS_TOP, vlm_mode=True, max_q_len=MAX_Q_LEN)
        
        # Loading pre-trained Word2Vec embeddings
        embed_matrix = np.zeros((vocab_size, EMBED_DIM))
        if os.path.exists(W2V_PATH):
            print("Loading Word2Vec...")
            from gensim.models import KeyedVectors
            w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
            for w, i in word_index.items():
                if w in w2v: embed_matrix[i] = w2v[w]
        else:
            print("warning: using random embeddings")
            
    else:
        X_images, Y_ids, Y_one_hot, filenames, questions_text, top_ans = \
            load_and_preprocess_data(JSON_PATH, IMAGE_PATH, IMG_SIZE, NUM_ANS_TOP)
        
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    all_true, all_pred = [], []
    all_filenames, all_questions = [], []
    inference_times = []
    histories = []

    print(f"\n--- Starting {model_type.upper()} CV ---")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_images, Y_ids), start=1):
        print(f"\nFold {fold}/{N_FOLDS}...")
        K.clear_session()

        Xi_tr, Xi_val = X_images[train_idx], X_images[val_idx]
        Y_tr, Y_val = Y_one_hot[train_idx], Y_one_hot[val_idx]
        val_fnames = filenames[val_idx]
        val_qs = np.array(questions_text)[val_idx]
        
        es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        
        if VNM_MODE:
            Xt_tr, Xt_val = X_text[train_idx], X_text[val_idx]
            model = build_vlm_model(vocab_size, embed_matrix, MAX_Q_LEN, EMBED_DIM)
            h = model.fit([Xi_tr, Xt_tr], Y_tr, validation_data=([Xi_val, Xt_val], Y_val),
                      epochs=30, batch_size=32, callbacks=[es], verbose=1)
            # Warm-up step
            _ = model.predict([Xi_val[:1], Xt_val[:1]], verbose=0)
            start_time = time.time()
            preds_probs = model.predict([Xi_val, Xt_val], verbose=0)
        else:
            model = build_cnn_baseline()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            h = model.fit(Xi_tr, Y_tr, validation_data=(Xi_val, Y_val),
                      epochs=25, batch_size=32, callbacks=[es], verbose=1)
            # Warm-up step
            _ = model.predict(Xi_val[:1], verbose=0)
            start_time = time.time()
            preds_probs = model.predict(Xi_val, verbose=0)

        end_time = time.time()
        avg_time = ((end_time - start_time) / len(Xi_val)) * 1000
        inference_times.append(avg_time)

        histories.append(h.history)
        preds = np.argmax(preds_probs, axis=1)
        trues = np.argmax(Y_val, axis=1)

        all_pred.extend(preds)
        all_true.extend(trues)
        all_filenames.extend(val_fnames)
        all_questions.extend(val_qs)

    # --- SAVING ---
    model_params = model.count_params()
    global_acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average='macro')
    mean_inference_time = np.mean(inference_times)

    is_closed = np.array([top_ans[i] in ['yes', 'no'] for i in all_true])
    acc_closed = accuracy_score(np.array(all_true)[is_closed], np.array(all_pred)[is_closed])
    acc_open = accuracy_score(np.array(all_true)[~is_closed], np.array(all_pred)[~is_closed])

    results_data = {
        'model': 'VLM_SOTA' if VNM_MODE else 'CNN_Baseline',
        'accuracy': global_acc,
        'macro_f1': macro_f1,
        'acc_closed': acc_closed,
        'acc_open': acc_open,
        'all_true': all_true,
        'all_pred': all_pred,
        'classes': top_ans,
        'filenames': all_filenames,
        'questions': all_questions,
        'history': histories,
        'inference_time_ms': mean_inference_time,
        'params': model_params
    }

    save_file = os.path.join(RESULTS_PATH, 'vlm_results.pkl' if VNM_MODE else 'cnn_results.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(results_data, f)

    print(f"\n Results saved to: {save_file}")
    print(f"\n Average Inference Time: {mean_inference_time:.2f} ms/sample")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Med-VQA models")
    parser.add_argument('--model_type', type=str, choices=['vlm', 'baseline'], required=True, help="Model type to train")
    args = parser.parse_args()
    train(args.model_type)

