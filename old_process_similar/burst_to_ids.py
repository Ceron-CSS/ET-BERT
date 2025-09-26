#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from tqdm import tqdm
import multiprocessing as mp
from typing import Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 搜索路径
sys.path.append(project_root)

from uer.utils import *

class Args:
    """简单的参数类，用于初始化tokenizer"""
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.spm_model_path = None
        self.tgt_vocab_path = None
        self.tgt_spm_model_path = None


def _worker_process(input_file: str, output_part_file: str, vocab_path: str, record_start_idx: int, record_end_idx: int) -> Tuple[int, int]:
    """
    单个进程的工作函数：以空行作为记录分隔，处理记录索引区间
    [record_start_idx, record_end_idx) 的完整记录，并写入分片文件。

    返回值: (processed_records, scanned_records_in_chunk)
    """
    # 初始化tokenizer（每个进程各自初始化，避免进程间对象共享问题）
    args = Args(vocab_path)
    tokenizer = str2tokenizer["bert"](args)

    processed_records = 0
    scanned_records = 0

    def flush_buffer(buffer, outfile):
        nonlocal processed_records
        if not buffer:
            return
        try:
            combined_text = " ".join(buffer)
            tokens = tokenizer.tokenize(combined_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids_str = " ".join(map(str, token_ids))
            outfile.write(token_ids_str + "\n")
        except Exception:
            outfile.write("\n")
        processed_records += 1

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_part_file, 'w', encoding='utf-8') as outfile:
        buffer = []
        current_record_idx = 0
        for line in infile:
            s = line.strip()
            if s == "":
                # 一个记录结束
                if current_record_idx >= record_start_idx and current_record_idx < record_end_idx:
                    flush_buffer(buffer, outfile)
                # 进入下一条记录
                if buffer:
                    scanned_records += 1
                    current_record_idx += 1
                buffer = []
                # 若已经达到需要的结束记录索引，继续清空直到找到下一个分隔即可结束
                if current_record_idx >= record_end_idx:
                    continue
            else:
                buffer.append(s)

        # EOF 情况：文件结尾可能没有空行，需要处理最后一个缓冲作为一条记录
        if buffer:
            if current_record_idx >= record_start_idx and current_record_idx < record_end_idx:
                flush_buffer(buffer, outfile)
            scanned_records += 1
            current_record_idx += 1

    return processed_records, scanned_records


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

        processed_records = 0
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            buffer = []
            for line in tqdm(infile, total=total_lines, desc="处理进度"):
                s = line.strip()
                if s == "":
                    # 边界：输出上一条记录（若有）
                    if buffer:
                        try:
                            combined_text = " ".join(buffer)
                            tokens = tokenizer.tokenize(combined_text)
                            token_ids = tokenizer.convert_tokens_to_ids(tokens)
                            token_ids_str = " ".join(map(str, token_ids))
                            outfile.write(token_ids_str + "\n")
                        except Exception:
                            outfile.write("\n")
                        processed_records += 1
                        buffer = []
                    # 空行本身跳过（不输出）
                    continue
                else:
                    buffer.append(s)

            # EOF 后，如有未刷新的记录，输出
            if buffer:
                try:
                    combined_text = " ".join(buffer)
                    tokens = tokenizer.tokenize(combined_text)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    token_ids_str = " ".join(map(str, token_ids))
                    outfile.write(token_ids_str + "\n")
                except Exception:
                    outfile.write("\n")
                processed_records += 1

        print(f"处理完成！共处理了 {processed_records} 条记录")
        print(f"结果已保存到: {output_file}")
        return

    # 多进程路径
    print(f"开始处理文件...(多进程: {processes_num})")
    # 先扫描记录总数（以空行分隔的段落）
    total_records = 0
    with open(input_file, 'r', encoding='utf-8') as infile:
        has_content = False
        for line in infile:
            if line.strip() == "":
                if has_content:
                    total_records += 1
                    has_content = False
            else:
                has_content = True
        if has_content:
            total_records += 1

    if total_records == 0:
        # 没有记录，直接创建空文件
        open(output_file, 'w', encoding='utf-8').close()
        print("未发现记录，已输出空文件。")
        return

    # 按记录数量切分到各进程
    records_per_proc = (total_records + processes_num - 1) // processes_num
    ranges = []
    rs = 0
    for i in range(processes_num):
        re = min(rs + records_per_proc, total_records)
        if rs >= re:
            break
        ranges.append((i, rs, re))
        rs = re

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
    print(f"处理完成！共处理了 {total_processed} 条记录")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='处理burst文件，转换为token ID序列')
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
                       default=16,
                       help='并行进程数，默认16为进程')
    
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
