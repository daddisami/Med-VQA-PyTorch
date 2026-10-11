import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import argparse
from config import *

def evaluate(model_type, results_path):
    """
    Evaluates a given model based on its saved results.
    """
    save_file = os.path.join(results_path, 'vlm_results.pkl' if model_type == 'vlm' else 'cnn_results.pkl')
    
    if not os.path.exists(save_file):
        print(f"Error: Results file not found at {save_file}")
        return

    with open(save_file, 'rb') as f:
        results_data = pickle.load(f)

    print(f"\n--- Evaluation Results for {results_data['model']} ---")
    print(f"Overall Accuracy: {results_data['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score: {results_data['macro_f1']:.4f}")
    print(f"Closed-Ended Questions Accuracy: {results_data['acc_closed'] * 100:.2f}%")
    print(f"Open-Ended Questions Accuracy: {results_data['acc_open'] * 100:.2f}%")
    print(f"Average Inference Time: {results_data['inference_time_ms']:.2f} ms/sample")
    print(f"Model Parameters: {results_data['params']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Med-VQA models")
    parser.add_argument('--model_type', type=str, choices=['vlm', 'baseline'], required=True, help="Model type to evaluate")
    parser.add_argument('--results_dir', type=str, default=RESULTS_PATH, help="Directory containing results files")
    args = parser.parse_args()
    evaluate(args.model_type, args.results_dir)

