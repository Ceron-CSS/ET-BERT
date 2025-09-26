#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
from collections import Counter, defaultdict
from typing import Dict, List


def build_cooccurrence(input_file: str, output_pickle: str, window_size: int = 5) -> None:
    """
    构建基于窗口的词共现统计：
    - 对每一行，按空格分割为token序列；不跨行。
    - 对序列中每个位置i，以自身为中心、总长window_size(奇数)的窗口，
      将窗口内除自身外的词计入 cooc[center][context] += 1。
    - 边界不足时使用现有范围，不做填充。
    - 结果以 {str -> Counter} 结构保存为pickle。
    """
    assert window_size % 2 == 1, "window_size 必须为奇数"
    half = window_size // 2

    cooc: Dict[str, Counter] = defaultdict(Counter)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens: List[str] = line.split()
            n = len(tokens)
            if n == 0:
                continue
            for i, center in enumerate(tokens):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                # 累计上下文（不含自身）
                for j in range(start, end):
                    if j == i:
                        continue
                    context = tokens[j]
                    cooc[center][context] += 1

    # 保存
    out_dir = os.path.dirname(output_pickle)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_pickle, 'wb') as pf:
        pickle.dump(cooc, pf, protocol=pickle.HIGHEST_PROTOCOL)


def query_top_k(pickle_file: str, word: str, top_k: int = 3) -> List[str]:
    with open(pickle_file, 'rb') as pf:
        cooc: Dict[str, Counter] = pickle.load(pf)
    cnt = cooc.get(word)
    if not cnt:
        return []
    return [f"{w}\t{c}" for w, c in cnt.most_common(top_k)]


def main():
    parser = argparse.ArgumentParser(description='窗口共现统计与查询（pickle保存）')
    parser.add_argument('--input_file', type=str, help='输入ID序列文件（每行空格分隔）')
    parser.add_argument('--output_pickle', type=str, help='输出pickle路径')
    parser.add_argument('--window_size', type=int, default=5, help='窗口大小（奇数），默认5')
    parser.add_argument('--build', action='store_true', help='构建共现统计并保存')
    parser.add_argument('--query', type=str, default='', help='要查询的词（字符串形式）')
    parser.add_argument('--top', type=int, default=3, help='查询Top-K，默认3')
    parser.add_argument('--pickle_file', type=str, help='已构建的pickle，用于查询')

    args = parser.parse_args()

    if args.build:
        assert args.input_file and args.output_pickle, '--build 需要 --input_file 与 --output_pickle'
        build_cooccurrence(args.input_file, args.output_pickle, args.window_size)
        print(f"共现统计已保存到: {args.output_pickle}")

    if args.query:
        pf = args.pickle_file or args.output_pickle
        assert pf, '查询需要提供 --pickle_file 或在同一命令中指定 --output_pickle（配合 --build）'
        top_list = query_top_k(pf, args.query, args.top)
        if not top_list:
            print('未找到该词或无共现统计。')
        else:
            print(f"词 '{args.query}' 的Top-{args.top} 共现：")
            for i, row in enumerate(top_list, 1):
                print(f"{i}.  {row}")


if __name__ == '__main__':
    main()


