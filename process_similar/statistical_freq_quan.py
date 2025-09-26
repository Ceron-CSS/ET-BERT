#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np


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


def _read_vocab_size(vocab_file: str) -> int:
    """
    读取词表大小：按行计数。假设每一行表示一个词/ID。
    """
    with open(vocab_file, 'r', encoding='utf-8') as vf:
        return sum(1 for _ in vf)


def build_cooccurrence_memmap(input_file: str,
                              vocab_file: str,
                              output_memmap: str,
                              window_size: int = 5,
                              dtype: str = 'uint32') -> Tuple[int, str]:
    """
    基于窗口统计，构建稠密共现矩阵（包含中心词自身），存为 memmap：
    - 先从 vocab 文件推断词表大小 V
    - 建立形状为 (V, V) 的内存映射文件，类型为无符号整型计数
    - 对输入的每一行（空格分隔的整型ID），以 window_size 为窗口，
      对于位置 i 的中心词 c，窗口内每个词 w（包含 c 本身）执行 M[c, w] += 1
    返回 (V, 输出路径)
    """
    assert window_size % 2 == 1, "window_size 必须为奇数"
    half = window_size // 2

    vocab_size = _read_vocab_size(vocab_file)
    out_dir = os.path.dirname(output_memmap)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mm = np.memmap(output_memmap, mode='w+', dtype=dtype, shape=(vocab_size, vocab_size))
    mm[:] = 0

    invalid_token_count = 0
    total_token_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 期望为ID序列
            try:
                tokens = [int(x) for x in line.split()]
            except ValueError:
                # 行中存在非整型，跳过本行
                continue

            n = len(tokens)
            if n == 0:
                continue
            total_token_count += n

            for i, center in enumerate(tokens):
                if not (0 <= center < vocab_size):
                    invalid_token_count += 1
                    continue
                start = max(0, i - half)
                end = min(n, i + half + 1)
                # 包含自身
                for j in range(start, end):
                    ctx = tokens[j]
                    if 0 <= ctx < vocab_size:
                        mm[center, ctx] += 1
                    else:
                        invalid_token_count += 1

    mm.flush()
    if invalid_token_count:
        print(f"警告：遇到 {invalid_token_count} 个越界或非法token（总读取 {total_token_count}）。", file=sys.stderr)
    return vocab_size, output_memmap


def query_top_k_memmap(memmap_file: str, row_index: int, top_k: int = 3) -> List[str]:
    """
    查询 memmap 矩阵某一行的 Top-K 共现（返回 "col\tcount" 字符串列表）。
    """
    mm = np.memmap(memmap_file, mode='r', dtype='uint32')
    # 推断维度：是平方矩阵
    size = mm.size
    vocab_size = int(np.sqrt(size))
    assert vocab_size * vocab_size == size, "memmap 文件大小与平方矩阵不匹配"
    mm = mm.reshape((vocab_size, vocab_size))
    assert 0 <= row_index < vocab_size, "row_index 越界"

    row = mm[row_index]
    if top_k <= 0:
        return []
    if np.all(row == 0):
        return []
    # argsort 得到从小到大，取尾部
    top_indices = np.argpartition(-row, kth=min(top_k, row.size - 1))[:top_k]
    # 再按值排序
    top_indices = top_indices[np.argsort(-row[top_indices])]
    return [f"{int(idx)}\t{int(row[idx])}" for idx in top_indices]


def export_memmap_row_to_tsv(memmap_file: str, row_index: int, out_tsv: str) -> None:
    """
    将 memmap 矩阵中指定行导出为 TSV：两列（col\tcount），导出整行（包含0）。
    """
    mm = np.memmap(memmap_file, mode='r', dtype='uint32')
    size = mm.size
    vocab_size = int(np.sqrt(size))
    assert vocab_size * vocab_size == size, "memmap 文件大小与平方矩阵不匹配"
    mm = mm.reshape((vocab_size, vocab_size))
    assert 0 <= row_index < vocab_size, "row_index 越界"

    row = mm[row_index]
    out_dir = os.path.dirname(out_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_tsv, 'w', encoding='utf-8') as fw:
        for c in range(vocab_size):
            fw.write(f"{c}\t{int(row[c])}\n")


def query_top_k(pickle_file: str, word: str, top_k: int = 3) -> List[str]:
    with open(pickle_file, 'rb') as pf:
        cooc: Dict[str, Counter] = pickle.load(pf)
    cnt = cooc.get(word)
    if not cnt:
        return []
    return [f"{w}\t{c}" for w, c in cnt.most_common(top_k)]


def main():
    parser = argparse.ArgumentParser(description='窗口共现统计与查询（支持 memmap 稠密矩阵 与 pickle 稀疏结构）')
    parser.add_argument('--input_file', type=str, help='输入ID序列文件（每行空格分隔）')
    parser.add_argument('--window_size', type=int, default=5, help='窗口大小（奇数），默认5')
    parser.add_argument('--build', action='store_true', help='构建共现统计并保存')

    # 稠密矩阵（memmap）参数
    parser.add_argument('--vocab_file', type=str, help='vocab 文件路径，用于推断词表大小')
    parser.add_argument('--output_memmap', type=str, help='输出 memmap 文件路径（.dat等）')
    parser.add_argument('--query_idx', type=int, help='查询的行索引（整型ID）')
    parser.add_argument('--memmap_file', type=str, help='已构建的 memmap 文件路径')
    parser.add_argument('--export_row_tsv', type=int, help='将指定行导出为 TSV（仅count>0），传入行索引')
    parser.add_argument('--tsv_out', type=str, help='导出 TSV 的输出路径')

    # 兼容旧版（pickle 稀疏结构）
    parser.add_argument('--output_pickle', type=str, help='输出pickle路径（旧版兼容）')
    parser.add_argument('--query', type=str, default='', help='要查询的词（字符串形式，旧版兼容）')
    parser.add_argument('--top', type=int, default=3, help='查询Top-K，默认3')
    parser.add_argument('--pickle_file', type=str, help='已构建的pickle，用于查询（旧版兼容）')

    args = parser.parse_args()

    if args.build:
        # 优先构建稠密矩阵
        if args.vocab_file and args.output_memmap:
            assert args.input_file, '--build 需要 --input_file'
            vsize, path = build_cooccurrence_memmap(
                input_file=args.input_file,
                vocab_file=args.vocab_file,
                output_memmap=args.output_memmap,
                window_size=args.window_size,
            )
            print(f"memmap 稠密矩阵已保存到: {path}，V={vsize}")
        else:
            # 回退到旧版 pickle 稀疏结构
            assert args.input_file and args.output_pickle, '--build 需要 --input_file 与 --output_pickle（或提供 --vocab_file 与 --output_memmap）'
            build_cooccurrence(args.input_file, args.output_pickle, args.window_size)
            print(f"共现统计已保存到: {args.output_pickle}")

    if args.query_idx is not None:
        mf = args.memmap_file or args.output_memmap
        assert mf, '查询需要提供 --memmap_file 或在同一命令中指定 --output_memmap（配合 --build）'
        top_list = query_top_k_memmap(mf, int(args.query_idx), args.top)
        if not top_list:
            print('该行没有共现或行索引越界/无效。')
        else:
            print(f"行 {args.query_idx} 的Top-{args.top} 共现列与计数：")
            for i, row in enumerate(top_list, 1):
                print(f"{i}.  {row}")

    if args.export_row_tsv is not None:
        mf = args.memmap_file or args.output_memmap
        assert mf, '导出需要提供 --memmap_file 或在同一命令中指定 --output_memmap（配合 --build）'
        assert args.tsv_out, '导出 TSV 需要指定 --tsv_out 输出路径'
        export_memmap_row_to_tsv(mf, int(args.export_row_tsv), args.tsv_out)
        print(f"已导出行 {args.export_row_tsv} 到 TSV: {args.tsv_out}")

    if args.query:
        # 旧版兼容查询
        pf = args.pickle_file or args.output_pickle
        if pf:
            top_list = query_top_k(pf, args.query, args.top)
            if not top_list:
                print('未找到该词或无共现统计。')
            else:
                print(f"词 '{args.query}' 的Top-{args.top} 共现：")
                for i, row in enumerate(top_list, 1):
                    print(f"{i}.  {row}")


if __name__ == '__main__':
    main()


