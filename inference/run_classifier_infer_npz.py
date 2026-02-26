"""
This script provides an example to wrap UER-py for classification inference.
- 输出 prediction_path(txt)
- 额外保存 npz（包含你其他方法一致的字段）：
  model_path, dataset, splitrate, labels, probabilities, predictions, class_names
"""

import sys
import os
import torch
import argparse
import torch.nn as nn
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

# 允许直接导入 fine-tuning 目录下的 run_classifier.py
fine_tuning_dir = os.path.join(uer_dir, "fine-tuning")
if os.path.isdir(fine_tuning_dir) and fine_tuning_dir not in sys.path:
    sys.path.append(fine_tuning_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts
from run_classifier import Classifier


def batch_loader(batch_size, src, seg):
    instances_num = src.size(0)
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, seg_batch
    if instances_num > (instances_num // batch_size) * batch_size:
        src_batch = src[(instances_num // batch_size) * batch_size:, :]
        seg_batch = seg[(instances_num // batch_size) * batch_size:, :]
        yield src_batch, seg_batch


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                line = line.strip().split("\t")
                for i, column_name in enumerate(line):
                    columns[column_name] = i
                continue

            line = line.strip().split("\t")
            if not line or ("text_a" not in columns):
                continue

            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence pair classification.
                text_a = line[columns["text_a"]]
                text_b = line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                )
                src_b = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
                )
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]

            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)

            dataset.append((src, seg))

    return dataset


def read_gold_labels(eva_path):
    """从 eva_path 读取真实标签，返回 np.int64 的 labels 数组"""
    gold_labels = []
    with open(eva_path, mode="r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        columns = {name: i for i, name in enumerate(header)}
        if "label" not in columns:
            raise ValueError(f"'label' column not found in eva file: {eva_path}, header={header}")

        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            gold_labels.append(int(parts[columns["label"]]))
    return np.array(gold_labels, dtype=np.int64)


def load_class_names(labels_num, class_names_path=""):
    """
    读取类别名（每行一个），否则默认 ['0','1',...]
    返回 dtype=object 的 np.array
    """
    if class_names_path:
        names = []
        with open(class_names_path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    names.append(t)
        if len(names) != labels_num:
            raise ValueError(f"class_names length mismatch: got {len(names)} but labels_num={labels_num}")
        return np.array(names, dtype=object)

    return np.array([str(i) for i in range(labels_num)], dtype=object)


def calculate_metrics_from_arrays(gold_labels, pred_labels, labels_num):
    """直接用数组计算混淆矩阵/指标"""
    print("Starting Post-Inference Evaluation...")

    if len(gold_labels) != len(pred_labels):
        print(f"Error: Length mismatch! Gold: {len(gold_labels)}, Pred: {len(pred_labels)}")
        return

    confusion = torch.zeros(labels_num, labels_num, dtype=torch.long)
    for p, g in zip(pred_labels, gold_labels):
        confusion[int(p), int(g)] += 1

    print("Confusion matrix:\n", confusion)
    print("-" * 30)
    
    eps = 1e-9
    correct = 0
    # 用于存储每个类别的指标，方便后续计算宏平均
    all_p, all_r, all_f1 = [], [], []

    for i in range(labels_num):
        tp = confusion[i, i].item()
        fp = confusion[i, :].sum().item() - tp
        fn = confusion[:, i].sum().item() - tp
        
        # 计算 Precision, Recall, F1
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)
        correct += tp
        
        print(f"Label {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    # --- 计算总体指标 ---
    accuracy = correct / len(gold_labels)
    # 宏平均 (Macro Average): 每个类别的指标简单相加除以类别数
    macro_p = sum(all_p) / labels_num
    macro_r = sum(all_r) / labels_num
    macro_f1 = sum(all_f1) / labels_num

    print("-" * 30)
    print(f"OVERALL METRICS:")
    print(f"Acc. (Correct/Total): {accuracy:.4f} ({correct}/{len(gold_labels)})")
    print(f"Macro Precision:      {macro_p:.4f}")
    print(f"Macro Recall:         {macro_r:.4f}")
    print(f"Macro F1-score:       {macro_f1:.4f}")
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion": confusion
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    infer_opts(parser)

    # 你想在 npz 里保存的额外信息（如果你不传，就给默认值）
    parser.add_argument("--dataset", type=str, default="",
                        help="Optional. Dataset name/path to store into npz. If empty, will use test_path.")
    parser.add_argument("--splitrate", type=float, default=-1.0,
                        help="Optional. Split rate used when building dataset. If unknown, keep -1.")
    parser.add_argument("--class_names_path", type=str, default="",
                        help="Optional. A txt file with one class name per line. Length must equal labels_num.")

    parser.add_argument("--save_npz", type=str, default="",
                        help="If set, save predictions/logits/prob/labels to a compressed npz file.")

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    clf = Classifier(args)
    clf = load_model(clf, args.load_model_path)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = clf.to(device)
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs are available. Let's use them.")
        clf = torch.nn.DataParallel(clf)

    # read data
    dataset = read_dataset(args, args.test_path)
    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size(0)
    print("The number of prediction instances: ", instances_num)

    clf.eval()

    # 真实标签（用于 npz 保存 + 评估）
    gold_labels = read_gold_labels(args.eva_path)  # [N]
    if len(gold_labels) != instances_num:
        print(f"⚠️ Warning: eva labels num ({len(gold_labels)}) != test instances ({instances_num}). "
              f"Evaluation may be invalid.")
    # 类别名
    class_names = load_class_names(args.labels_num, args.class_names_path)

    # 收集推理结果
    all_preds = []
    all_logits = []
    all_probs = []

    # 输出 txt
    os.makedirs(os.path.dirname(args.prediction_path), exist_ok=True) if os.path.dirname(args.prediction_path) else None
    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        if args.output_logits:
            f.write("\tlogits")
        if args.output_prob:
            f.write("\tprob")
        f.write("\n")

        for (src_batch, seg_batch) in batch_loader(batch_size, src, seg):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)

            with torch.no_grad():
                _, logits = clf(src_batch, None, seg_batch)   # logits: [B, C]

            prob = nn.Softmax(dim=1)(logits)                 # [B, C]
            pred = torch.argmax(logits, dim=1)               # [B]

            # 收集（用于 npz）
            all_preds.append(pred.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_probs.append(prob.detach().cpu().numpy())

            # txt 输出
            pred_list = pred.detach().cpu().numpy().tolist()
            logits_list = logits.detach().cpu().numpy().tolist()
            prob_list = prob.detach().cpu().numpy().tolist()

            for j in range(len(pred_list)):
                f.write(str(pred_list[j]))
                if args.output_logits:
                    f.write("\t" + " ".join([str(v) for v in logits_list[j]]))
                if args.output_prob:
                    f.write("\t" + " ".join([str(v) for v in prob_list[j]]))
                f.write("\n")

    # 拼接成完整数组
    predictions = np.concatenate(all_preds, axis=0).astype(np.int64)        # [N]
    logits_np = np.concatenate(all_logits, axis=0).astype(np.float32)       # [N, C]
    probabilities = np.concatenate(all_probs, axis=0).astype(np.float32)    # [N, C]

    # ---- 保存 npz（字段与您其他方法一致）----
    if args.save_npz:
        out_dir = os.path.dirname(args.save_npz)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        dataset_str = args.dataset if args.dataset else str(args.test_path)

        np.savez_compressed(
            args.save_npz,
            model_path=str(args.load_model_path),
            dataset=str(dataset_str),
            splitrate=float(args.splitrate),
            labels=gold_labels.astype(np.int64),
            probabilities=probabilities,
            predictions=predictions,
            class_names=np.array(list(class_names), dtype=object),
        )
        print(f"Saved npz to: {args.save_npz}")

    # ---- 推理结束，执行评估（直接用数组，不再读文件）----
    calculate_metrics_from_arrays(gold_labels, predictions, args.labels_num)


if __name__ == "__main__":
    main()