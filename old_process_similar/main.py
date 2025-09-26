#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import shutil
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def run(cmd: list) -> None:
    print('> ' + ' '.join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def main():
    parser = argparse.ArgumentParser(description='流水线：burst -> ids -> 共现 -> 相似度(预计算)')
    parser.add_argument('--input_file', type=str, required=True, help='burst原始文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词表路径')
    parser.add_argument('--processes_num', type=int, default=12, help='并行进程数')
    parser.add_argument('--window_size', type=int, default=5, help='共现窗口大小(奇数)')
    parser.add_argument('--top_contexts', type=int, default=2000, help='构建向量的全局Top-N上下文')
    parser.add_argument('--top_k', type=int, default=10, help='保存每个词Top-K相似词')

    args = parser.parse_args()

    # 解析路径与文件名
    input_abs = os.path.abspath(args.input_file)
    vocab_abs = os.path.abspath(args.vocab_path)
    if not os.path.exists(input_abs):
        raise FileNotFoundError(f'输入不存在: {input_abs}')
    if not os.path.exists(vocab_abs):
        raise FileNotFoundError(f'词表不存在: {vocab_abs}')

    in_dir, in_name = os.path.split(input_abs)
    stem, _ = os.path.splitext(in_name)

    # 输出目录
    out_json_dir = os.path.join(PROJECT_ROOT, 'result', 'json')
    ensure_dir(out_json_dir)

    # 临时目录
    tmp_dir = os.path.join(PROJECT_ROOT, 'result', 'tmp')
    ensure_dir(tmp_dir)

    # 中间文件
    ids_path = os.path.join(tmp_dir, f'{stem}_ids.txt')
    pkl_path = os.path.join(tmp_dir, f'{stem}.pkl')
    neighbors_json = os.path.join(out_json_dir, f'{stem}.json')

    # 1) burst -> ids
    burst_to_ids = os.path.join(SCRIPT_DIR, 'burst_to_ids.py')
    run([
        sys.executable, burst_to_ids,
        '--input_file', input_abs,
        '--output_file', ids_path,
        '--vocab_path', vocab_abs,
        '--processes_num', str(args.processes_num),
    ])

    # 2) ids -> cooc pickle
    statistical_freq = os.path.join(SCRIPT_DIR, 'statistical_freq.py')
    run([
        sys.executable, statistical_freq,
        '--build',
        '--input_file', ids_path,
        '--output_pickle', pkl_path,
        '--window_size', str(args.window_size),
    ])

    # 3) cooc -> neighbors json（预计算Top-K）
    save_similarity = os.path.join(SCRIPT_DIR, 'save_similarity.py')
    run([
        sys.executable, save_similarity,
        '--pickle_file', pkl_path,
        '--top_contexts', str(args.top_contexts),
        '--top_k', str(args.top_k),
        '--precompute',
        '--save_neighbors', neighbors_json,
    ])

    # 清理中间文件
    try:
        if os.path.exists(ids_path):
            os.remove(ids_path)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
        # 保留 result/json 下的最终 json
    except OSError:
        pass

    print(f'完成。相似词JSON输出: {neighbors_json}')


if __name__ == '__main__':
    main()
