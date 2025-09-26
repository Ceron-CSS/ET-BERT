#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
from typing import List, Tuple


def load_neighbors(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data: { center: [[neighbor, score], ...] }
    return {k: [(w, float(s)) for w, s in v] for k, v in data.items()}


def query(neighbors_path: str, word: str, top_k: int) -> List[Tuple[str, float]]:
    neighbors = load_neighbors(neighbors_path)
    rows = neighbors.get(word, [])
    return rows[:top_k]


def main():
    parser = argparse.ArgumentParser(description='从已预计算的邻居JSON中查询某词的Top-K相似词')
    parser.add_argument('--neighbors_file', type=str, required=True, help='预计算邻居JSON文件路径')
    parser.add_argument('--query', type=str, required=True, help='查询的词（字符串）')
    parser.add_argument('--top_k', type=int, default=10, help='返回Top-K，默认10')

    args = parser.parse_args()

    rows = query(args.neighbors_file, args.query, args.top_k)
    if not rows:
        print('查询词不存在或无邻居。')
        return
    print(f"与 '{args.query}' 最相似的Top-{min(args.top_k, len(rows))}（来自预计算）:")
    for i, (w, s) in enumerate(rows, 1):
        print(f"{i}.  {w}\t{s:.6f}")

    # neighbors_path = "result/json/test_burst.json"
    # word = "38999"

    # top1 = query(neighbors_path, word, top_k=1)  # 返回形如 [(相似词, 分数)]
    # if top1:
    #     neighbor, score = top1[0]
    #     print(neighbor, score)
    # else:
    #     print("该词不存在或无邻居")


if __name__ == '__main__':
    main()


