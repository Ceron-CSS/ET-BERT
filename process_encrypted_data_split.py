#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from tqdm import tqdm
import multiprocessing as mp
from typing import Tuple

# 添加uer模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'uer'))

from uer.utils import *

class Args:
    """简单的参数类，用于初始化tokenizer"""
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.spm_model_path = None
        self.tgt_vocab_path = None
        self.tgt_spm_model_path = None


def _worker_process(input_file: str, output_part_file: str, vocab_path: str, start_idx: int, end_idx: int) -> Tuple[int, int]:
    """
    单个进程的工作函数：处理 [start_idx, end_idx) 行，并写入分片文件。

    返回值: (processed_lines, total_lines_in_chunk)
    """
    # 初始化tokenizer（每个进程各自初始化，避免进程间对象共享问题）
    args = Args(vocab_path)
    tokenizer = str2tokenizer["bert"](args)

    processed_lines = 0
    total_in_chunk = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_part_file, 'w', encoding='utf-8') as outfile:
        # 跳到起始行
        pos = 0
        while pos < start_idx:
            if not infile.readline():
                break
            pos += 1

        # 处理指定范围
        while pos < end_idx:
            line = infile.readline()
            if not line:
                break
            pos += 1
            total_in_chunk += 1

            line = line.strip()
            if not line:
                outfile.write("\n")
                continue

            try:
                tokens = tokenizer.tokenize(line)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_ids_str = " ".join(map(str, token_ids))
                outfile.write(token_ids_str + "\n")
                processed_lines += 1
            except Exception:
                outfile.write("\n")
                processed_lines += 1
                continue

    return processed_lines, total_in_chunk


def process_encrypted_file(input_file, output_file, vocab_path, processes_num=1):
    """
    处理加密文件，将每行内容转换为token ID序列
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径  
        vocab_path: 词汇表文件路径
    """
    print(f"正在处理文件: {input_file}")
    print(f"使用词汇表: {vocab_path}")
    
    # 统计文件行数
    print("正在统计文件行数...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"文件总行数: {total_lines}")

    if processes_num <= 1 or total_lines == 0:
        # 单进程路径
        print("开始处理文件...(单进程)")
        # 初始化tokenizer
        args = Args(vocab_path)
        tokenizer = str2tokenizer["bert"](args)
        print(f"词汇表大小: {len(tokenizer.vocab)}")

        processed_lines = 0
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            for line in tqdm(infile, total=total_lines, desc="处理进度"):
                line = line.strip()
                if not line:
                    outfile.write("\n")
                    continue
                try:
                    tokens = tokenizer.tokenize(line)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    token_ids_str = " ".join(map(str, token_ids))
                    outfile.write(token_ids_str + "\n")
                    processed_lines += 1
                except Exception:
                    outfile.write("\n")
                    processed_lines += 1
                    continue
        print(f"处理完成！共处理了 {processed_lines} 行")
        print(f"结果已保存到: {output_file}")
        return

    # 多进程路径
    print(f"开始处理文件...(多进程: {processes_num})")
    # 切分行范围
    lines_per_proc = (total_lines + processes_num - 1) // processes_num
    ranges = []
    start = 0
    for i in range(processes_num):
        end = min(start + lines_per_proc, total_lines)
        if start >= end:
            break
        ranges.append((i, start, end))
        start = end

    # 启动进程处理
    part_files = [f"{output_file}.part{i}" for i, _, _ in ranges]
    with mp.Pool(processes=len(ranges)) as pool:
        results = []
        for (i, s, e), part in zip(ranges, part_files):
            results.append(pool.apply_async(_worker_process, (input_file, part, vocab_path, s, e)))
        stats = [r.get() for r in tqdm(results, desc="子进程完成", total=len(results))]

    # 合并分片
    print("正在合并分片...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for part in part_files:
            with open(part, 'r', encoding='utf-8') as pf:
                for line in pf:
                    outfile.write(line)
            try:
                os.remove(part)
            except OSError:
                pass

    total_processed = sum(p for p, _ in stats)
    print(f"处理完成！共处理了 {total_processed} 行")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='处理encrypted_USTC_TFC_burst.txt文件，转换为token ID序列')
    parser.add_argument('--input_file', type=str, 
                       default='corpora/encrypted_USTC_TFC_burst.txt',
                       help='输入文件路径')
    parser.add_argument('--output_file', type=str,
                       default='corpora/ids/USTC_TFC_tokenized_ids.txt', 
                       help='输出文件路径')
    parser.add_argument('--vocab_path', type=str,
                       default='models/encryptd_vocab_USTC_TFC_all.txt',
                       help='词汇表文件路径')
    parser.add_argument('--processes_num', type=int,
                       default=1,
                       help='并行进程数，默认1为单进程')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 {args.input_file} 不存在")
        return
    
    # 检查词汇表文件是否存在
    if not os.path.exists(args.vocab_path):
        print(f"错误: 词汇表文件 {args.vocab_path} 不存在")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理文件
    process_encrypted_file(args.input_file, args.output_file, args.vocab_path, args.processes_num)


if __name__ == "__main__":
    main()
