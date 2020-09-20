import os
import re
import argparse
import numpy as np


def detect_accuracy(y_true, y_pred, thresh=0.8):
    true_cls = y_true[:,2]
    pred_cls = y_pred[:,2]
    pred_cls = (pred_cls > thresh).astype(np.float32)
    
    abs_err = np.abs(true_cls-pred_cls)
    return 1-np.mean(abs_err, axis=0)

def position_accuracy(y_true, y_pred):
    true_cls = y_true[:,2]
    true_pos = y_true[:,:2]
    pred_pos = y_pred[:,:2]
    
    abs_err = np.abs(true_pos-pred_pos)
    dist_sqr_err = np.sum(abs_err**2, axis=1)
    dist_err = dist_sqr_err**0.5
    
    # only consider when object is there
    dist_err = dist_err*true_cls
    net_err = np.sum(dist_err)
    total_objects = np.sum(true_cls)
    
    mean_err = net_err / max(total_objects, 1)
    return 1-mean_err     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("truth")
    parser.add_argument("pred")

    args = parser.parse_args()

    data_truth = {}
    data_pred = {}

    truth_labels = set([])
    pred_labels = set([])

    p = re.compile(r"sample_(\d+)\..+")

    print("loading truth") 
    with open(args.truth, "r") as fp:
        for i, line in enumerate(fp.readlines()):
            if i == 0:
                continue
            tokens = line.strip().split()
            if len(tokens) != 6:
                continue
            filename, x, y, _, _, confidence = tokens

            m = p.findall(filename)
            if len(m) == 0:
                continue
            iid = int(m[0])
            x = float(x)
            y = float(y)
            confidence = float(confidence)

            data_truth[iid] = (x, y, confidence)
            truth_labels.add(iid)
        
    print("loading pred") 
    p_pred = re.compile(r"sample_(\d+)")
    with open(args.pred, "r") as fp:
        for i, line in enumerate(fp.readlines()):
            if i == 0:
                continue
            tokens = line.strip().split()
            if len(tokens) != 4:
                continue
            filename, x, y, confidence = tokens

            m = p.findall(filename)
            if len(m) == 0:
                continue
            iid = int(m[0])
            x = float(x)
            y = float(y)
            confidence = float(confidence)

            data_pred[iid] = (x, y, confidence)
            pred_labels.add(iid)
    
    missing_labels = truth_labels.difference(pred_labels)

    print(f"missing={len(missing_labels)}")

    y_truth = []
    y_pred = []

    for iid, truth in data_truth.items():
        if iid not in data_pred:
            continue
        pred = data_pred[iid]

        y_truth.append(np.array(truth))
        y_pred.append(np.array(pred))
    
    y_truth = np.array(y_truth)
    y_pred = np.array(y_pred)

    y_err = (y_truth-y_pred)**2
    y_err = np.sum(y_err, axis=1)
    mse = np.mean(y_err, axis=0)

    detect_acc = detect_accuracy(y_truth, y_pred)
    position_acc = position_accuracy(y_truth, y_pred)

    print(f"mse={mse:.3f} detect_acc={detect_acc:.3f} position_acc={position_acc:.3f}")
    

if __name__ == '__main__':
    main()