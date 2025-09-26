#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
import numpy as np
import json
from collections import Counter
from typing import Dict, List, Tuple


def load_cooc(pickle_file: str) -> Dict[str, Counter]:
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def build_context_vocab(cooc: Dict[str, Counter], top_contexts: int) -> List[str]:
    global_ctx = Counter()
    for cnt in cooc.values():
        global_ctx.update(cnt)
    if top_contexts > 0:
        return [w for w, _ in global_ctx.most_common(top_contexts)]
    return list(global_ctx.keys())


def build_dense_matrix(cooc: Dict[str, Counter], context_vocab: List[str]) -> Tuple[np.ndarray, List[str]]:
    centers = list(cooc.keys())
    ctx2idx = {w: i for i, w in enumerate(context_vocab)}
    X = np.zeros((len(centers), len(context_vocab)), dtype=np.float32)
    for i, c in enumerate(centers):
        cnt = cooc[c]
        for w, v in cnt.items():
            j = ctx2idx.get(w)
            if j is not None:
                X[i, j] = float(v)
    # L2归一化
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms
    return X, centers


def query_similar(X: np.ndarray, centers: List[str], query_word: str, top_k: int) -> List[Tuple[str, float]]:
    try:
        idx = centers.index(query_word)
    except ValueError:
        return []
    q = X[idx:idx+1]  # shape (1, d)
    sims = (X @ q.T).ravel()  # 余弦相似度（因行已L2归一）
    sims[idx] = -1.0  # 排除自身
    top_idx = np.argsort(-sims)[:top_k]
    return [(centers[i], float(sims[i])) for i in top_idx]


def precompute_topk_all(X: np.ndarray, centers: List[str], top_k: int, batch_size: int = 1024) -> Dict[str, List[Tuple[str, float]]]:
    """
    预计算每个中心词的Top-K相似词（余弦，相当于点积，因已L2归一）。
    采用分批计算避免一次性构建N×N相似度矩阵。
    返回 {center: [(neighbor, sim), ...]}。
    """
    n = X.shape[0]
    results: Dict[str, List[Tuple[str, float]]] = {}
    XT = X.T
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims_block = X[start:end] @ XT  # shape (b, n)
        # 排除自身
        for row_idx in range(end - start):
            global_idx = start + row_idx
            sims_block[row_idx, global_idx] = -1.0
        # 选Top-K（不排序的快速选择）
        if top_k < n:
            part_idx = np.argpartition(-sims_block, kth=top_k-1, axis=1)[:, :top_k]
        else:
            part_idx = np.argsort(-sims_block, axis=1)
        # 排序这K个
        for row_idx in range(end - start):
            idxs = part_idx[row_idx]
            vals = sims_block[row_idx, idxs]
            order = np.argsort(-vals)
            top_ids = idxs[order]
            top_vals = vals[order]
            results[centers[start + row_idx]] = [(centers[j], float(top_vals[k])) for k, j in enumerate(top_ids)]
    return results


def save_neighbors(neighbors: Dict[str, List[Tuple[str, float]]], out_path: str, fmt: str = 'json') -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    if fmt == 'json':
        serializable = {k: [(w, float(s)) for (w, s) in v] for k, v in neighbors.items()}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False)
    else:
        with open(out_path, 'wb') as f:
            pickle.dump(neighbors, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_neighbors(path: str) -> Dict[str, List[Tuple[str, float]]]:
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # JSON读进来后已是列表形式
        return {k: [(w, float(s)) for w, s in v] for k, v in data.items()}
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='基于共现向量的余弦相似度：查询或预计算邻居')
    parser.add_argument('--pickle_file', type=str, required=True, help='process_encrypted_cooc_stats.py 生成的 cooc.pkl')
    parser.add_argument('--top_contexts', type=int, default=5000, help='使用全局Top-N上下文构建向量，默认5000')
    parser.add_argument('--query', type=str, default='', help='查询的中心词字符串')
    parser.add_argument('--top_k', type=int, default=5, help='返回Top-K相似词，默认5')
    parser.add_argument('--neighbors_file', type=str, default='', help='已预计算邻居文件（json或pkl），用于直接查询')
    parser.add_argument('--precompute', action='store_true', help='预计算所有中心词的Top-K邻居并保存')
    parser.add_argument('--save_neighbors', type=str, default='', help='保存预计算邻居的路径(.json或.pkl)')
    parser.add_argument('--batch_size', type=int, default=1024, help='预计算时的批大小，默认1024')

    args = parser.parse_args()

    cooc = load_cooc(args.pickle_file)
    context_vocab = build_context_vocab(cooc, args.top_contexts)
    X, centers = build_dense_matrix(cooc, context_vocab)

    if args.precompute:
        assert args.save_neighbors, '--precompute 需要指定 --save_neighbors 输出路径'
        neighbors = precompute_topk_all(X, centers, args.top_k, args.batch_size)
        fmt = 'json' if args.save_neighbors.lower().endswith('.json') else 'pkl'
        save_neighbors(neighbors, args.save_neighbors, fmt)
        print(f"已预计算并保存所有中心词Top-{args.top_k}邻居到: {args.save_neighbors}")
        return

    if args.neighbors_file:
        neighbors = load_neighbors(args.neighbors_file)
        rows = neighbors.get(args.query, [])
        if not rows:
            print('查询词不存在或无预计算邻居。')
            return
        print(f"与 '{args.query}' 最相似的Top-{min(args.top_k, len(rows))}（来自预计算）:")
        for i, (w, s) in enumerate(rows[:args.top_k], 1):
            print(f"{i}. {w}\t{s:.6f}")
        return

    if args.query:
        results = query_similar(X, centers, args.query, args.top_k)
        if not results:
            print('查询词不存在或没有有效向量。')
            return
        print(f"与 '{args.query}' 最相似的Top-{args.top_k}：")
        for i, (w, s) in enumerate(results, 1):
            print(f"{i}. {w}\t{s:.6f}")


if __name__ == '__main__':
    main()
