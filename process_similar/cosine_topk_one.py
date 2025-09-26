#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import heapq
from typing import List, Tuple


def _infer_vocab_size(mm: np.memmap) -> int:
    size = mm.size
    v = int(np.sqrt(size))
    assert v * v == size, "memmap 文件大小与平方矩阵不匹配"
    return v


def topk_for_one_row(
    memmap_path: str,
    row_index: int,
    topk: int = 5,
    dtype: str = "uint32",
    ref_block: int = 2048,
    include_self: bool = False,
    eps: float = 1e-12,
) -> List[Tuple[int, float]]:
    """
    仅对指定行计算与所有行的余弦相似度，返回Top-K (index, score)。
    采用分块遍历参考行，避免一次性加载全部矩阵。
    """
    mm = np.memmap(memmap_path, mode="r", dtype=dtype)
    v = _infer_vocab_size(mm)
    mm = mm.reshape((v, v))

    assert 0 <= row_index < v, "row_index 越界"

    # 查询行向量与范数
    q = np.asarray(mm[row_index], dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm < eps:
        return []

    # 最小堆：存 (score, idx)
    heap: List[Tuple[float, int]] = []

    start = 0
    while start < v:
        end = min(start + ref_block, v)
        block = np.asarray(mm[start:end], dtype=np.float32)  # (B, V)

        dots = block @ q  # (B,)
        norms = np.linalg.norm(block, axis=1)  # (B,)
        sims = dots / (q_norm * norms + eps)

        for i, s in enumerate(sims):
            idx = start + i
            if not include_self and idx == row_index:
                continue
            if not np.isfinite(s):
                continue
            if len(heap) < topk:
                heapq.heappush(heap, (float(s), idx))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (float(s), idx))

        start = end

    # 从大到小输出 (idx, score)
    best = heapq.nlargest(topk, heap)
    return [(idx, score) for score, idx in best]


def main():
    parser = argparse.ArgumentParser(description="对稠密矩阵(memmap)的一行计算Top-K余弦近邻")
    parser.add_argument("--memmap", required=True, help="稠密矩阵 memmap 文件路径（如 choumi.dat）")
    parser.add_argument("--row_index", type=int, required=True, help="要查询的行索引")
    parser.add_argument("--output", required=True, help="输出文件路径（两列：索引\t相似度）")
    parser.add_argument("--topk", type=int, default=5, help="返回Top-K，默认5")
    parser.add_argument("--dtype", default="uint32", help="memmap 数据类型，默认uint32；可用uint16")
    parser.add_argument("--ref_block", type=int, default=2048, help="参考分块大小（行数），默认2048")
    parser.add_argument("--include_self", action="store_true", help="是否包含自身为候选（默认不包含）")

    args = parser.parse_args()

    results = topk_for_one_row(
        memmap_path=args.memmap,
        row_index=args.row_index,
        topk=args.topk,
        dtype=args.dtype,
        ref_block=args.ref_block,
        include_self=args.include_self,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for idx, score in results:
            f.write(f"{idx}\t{score:.8f}\n")


if __name__ == "__main__":
    main()

#python process_similar/cosine_topk_one.py --memmap result/dat/choumi.dat --row_index 38999 --topk 5 --output result/rows/38999_top5.tsv

