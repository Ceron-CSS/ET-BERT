#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np


def infer_vocab_size(mm: np.memmap) -> int:
    size = mm.size
    v = int(np.sqrt(size))
    assert v * v == size, "memmap 文件大小与平方矩阵不匹配"
    return v


def compute_row_norms(memmap_path: str, dtype: str = "uint32", chunk_rows: int = 1024) -> np.ndarray:
    mm = np.memmap(memmap_path, mode="r", dtype=dtype)
    v = infer_vocab_size(mm)
    mm = mm.reshape((v, v))
    norms = np.zeros((v,), dtype=np.float32)
    start = 0
    while start < v:
        end = min(start + chunk_rows, v)
        block = np.asarray(mm[start:end], dtype=np.float32)
        norms[start:end] = np.linalg.norm(block, axis=1)
        start = end
    return norms


def topk_all_rows(
    memmap_path: str,
    output_path: str,
    topk: int = 5,
    dtype: str = "uint32",
    query_block: int = 128,
    ref_block: int = 1024,
    include_self: bool = False,
    eps: float = 1e-12,
    output_format: str = "tsv",
) -> None:
    # 预读维度
    mm = np.memmap(memmap_path, mode="r", dtype=dtype)
    v = infer_vocab_size(mm)
    # 预计算范数
    norms = compute_row_norms(memmap_path, dtype=dtype, chunk_rows=max(1024, ref_block))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 如果需要二进制输出，则准备容器
    write_binary = output_format in {"npy", "npz"}
    if write_binary:
        all_neighbors = np.full((v, topk), -1, dtype=np.int32)
        all_scores = np.full((v, topk), -np.inf, dtype=np.float32)

    q_start = 0
    while q_start < v:
        q_end = min(q_start + query_block, v)
        Q = np.memmap(memmap_path, mode="r", dtype=dtype, shape=(v, v))[q_start:q_end]
        Q = np.asarray(Q, dtype=np.float32)  # (B, V)
        q_norms = norms[q_start:q_end]  # (B,)
        # 归一化查询
        with np.errstate(divide='ignore', invalid='ignore'):
            Qn = Q / (q_norms[:, None] + eps)

        # 维护该查询块的全局Top-K（向量化合并）
        best_scores = np.full((q_end - q_start, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((q_end - q_start, topk), -1, dtype=np.int32)

        r_start = 0
        while r_start < v:
            r_end = min(r_start + ref_block, v)
            R = np.memmap(memmap_path, mode="r", dtype=dtype, shape=(v, v))[r_start:r_end]
            R = np.asarray(R, dtype=np.float32)  # (C, V)
            r_norms = norms[r_start:r_end]  # (C,)
            with np.errstate(divide='ignore', invalid='ignore'):
                Rn = R / (r_norms[:, None] + eps)

            # 计算相似度 (B, C)
            sims = Qn @ Rn.T

            if not include_self:
                # 将重叠的对角位置置为 -inf
                overlap_start = max(q_start, r_start)
                overlap_end = min(q_end, r_end)
                if overlap_end > overlap_start:
                    rows = np.arange(overlap_start, overlap_end) - q_start
                    cols = np.arange(overlap_start, overlap_end) - r_start
                    sims[rows, cols] = -np.inf

            # 得到当前参考块内每行Top-K（局部）
            k_local = min(topk, sims.shape[1])
            idx_local_part = np.argpartition(sims, -k_local, axis=1)[:, -k_local:]
            scores_local = np.take_along_axis(sims, idx_local_part, axis=1)
            idx_local = idx_local_part + r_start

            # 合并局部Top-K与历史Top-K
            scores_cat = np.concatenate([best_scores, scores_local], axis=1)
            idx_cat = np.concatenate([best_idx, idx_local], axis=1)
            # 重新选出Top-K
            part = np.argpartition(scores_cat, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(scores_cat, part, axis=1)
            best_idx = np.take_along_axis(idx_cat, part, axis=1)

            r_start = r_end

        # 对该查询块按分数降序排序
        order = np.argsort(-best_scores, axis=1)
        best_scores = np.take_along_axis(best_scores, order, axis=1)
        best_idx = np.take_along_axis(best_idx, order, axis=1)

        if write_binary:
            all_neighbors[q_start:q_end] = best_idx
            all_scores[q_start:q_end] = best_scores
        else:
            # 逐行写出：row\tneighbor\tscore
            with open(output_path, "a", encoding="utf-8") as fw:
                for i in range(q_end - q_start):
                    row_idx = q_start + i
                    for j in range(topk):
                        s = best_scores[i, j]
                        idx = best_idx[i, j]
                        if idx < 0 or not np.isfinite(s):
                            continue
                        fw.write(f"{row_idx}\t{idx}\t{s:.8f}\n")

        q_start = q_end

    # 若为二进制输出，统一一次性保存
    if write_binary:
        if output_format == "npy":
            np.save(output_path + ".neighbors.npy", all_neighbors)
            np.save(output_path + ".scores.npy", all_scores)
        else:
            np.savez_compressed(output_path, neighbors=all_neighbors, scores=all_scores)


def main():
    parser = argparse.ArgumentParser(description="对稠密共现矩阵按行计算余弦相似度，输出每行Top-K最近邻（向量化加速版）")
    parser.add_argument("--memmap", required=True, help="稠密矩阵 memmap 文件路径（如 choumi.dat）")
    parser.add_argument("--output", required=True, help="输出路径（tsv直接写；npy/npz将生成对应文件）")
    parser.add_argument("--dtype", default="uint32", help="memmap 数据类型，默认uint32")
    parser.add_argument("--topk", type=int, default=5, help="每行Top-K，默认5")
    parser.add_argument("--query_block", type=int, default=128, help="查询块行数，默认128")
    parser.add_argument("--ref_block", type=int, default=1024, help="参考块行数，默认1024")
    parser.add_argument("--format", choices=["tsv", "npy", "npz"], default="tsv", help="输出格式，默认tsv")
    parser.add_argument("--include_self", action="store_true", help="是否包含自身为候选（默认不包含）")
    args = parser.parse_args()

    # 若为tsv，确保文件不存在或清空
    if args.format == "tsv" and os.path.exists(args.output):
        os.remove(args.output)

    topk_all_rows(
        memmap_path=args.memmap,
        output_path=args.output,
        topk=args.topk,
        dtype=args.dtype,
        query_block=args.query_block,
        ref_block=args.ref_block,
        include_self=args.include_self,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()

#python process_similar/cosine_top5_all.py --memmap result/dat/choumi.dat --output result/topk_all/top5_all.tsv

