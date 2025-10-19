import os
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    ref_labels = [json.loads(qs) for qs in open(os.path.expanduser(args.annotation_file), "r")]
    res_labels = [json.loads(qs) for qs in open(os.path.expanduser(args.result_file), "r")]

    results = {'TP': 0, 'TN': 0, 'FP':0, 'FN': 0}
    num_sample = len(ref_labels)

    for i, line in enumerate(tqdm(ref_labels)):
        idx = line["question_id"]
        assert idx == res_labels[i]['question_id']
        ref_label = line["label"].lower().strip()
        res_label = res_labels[i]["text"].lower().strip()

        if ref_label == 'yes':
            if 'yes' in res_label:
                results['TP'] += 1
            else:
                results['FN'] += 1
        else:
            if 'no' in res_label or 'not' in res_label:
                results['TN'] += 1
            else:
                results['FP'] += 1

    Accuracy = (results['TP'] + results['TN']) / num_sample        
    Precision = results['TP'] / (results['TP'] + results['FP'])
    Recall = results['TP'] / (results['TP'] + results['FN'])
    F1_score = 2 * Precision * Recall / (Precision + Recall)

    print('Accurancy', Accuracy)
    print('Precision', Precision)
    print('Recall', Recall)
    print('F1_score', F1_score)    
