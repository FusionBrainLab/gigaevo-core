#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from datasets import load_dataset


def process_narrativeqa_dataset(output_dir: str):
    """
    下载并保存 NarrativeQA 数据集的原始文件
    
    Args:
        output_dir: 输出目录路径，数据将保存到该目录下的 parquet 文件
    """
    print("=" * 60)
    print("下载 NarrativeQA 数据集")
    print("=" * 60)
    
    # 加载数据集
    print(f"\n正在加载数据集: deepmind/narrativeqa")
    ds = load_dataset("deepmind/narrativeqa")
    print(f"数据集加载成功！包含以下分割: {list(ds.keys())}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 保存每个分割的原始数据
    for split_name in ds.keys():
        print(f"\n{'='*60}")
        print(f"处理分割: {split_name}")
        print(f"{'='*60}")
        
        dataset = ds[split_name]
        print(f"样本数量: {len(dataset)}")
        
        # 查看数据集结构
        if len(dataset) > 0:
            print(f"数据集字段: {dataset.column_names}")
        
        # 直接保存原始数据为 parquet 文件
        print(f"\n正在保存原始数据为 parquet 文件...")
        output_path = os.path.join(output_dir, f"{split_name}.parquet")
        dataset.to_parquet(output_path)
        
        print(f"[OK] 已保存到: {output_path}")
        print(f"     文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    print(f"\n{'='*60}")
    print("所有分割处理完成！")
    print(f"{'='*60}")
    print(f"数据已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="下载并保存 NarrativeQA 数据集的原始文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/narrativeqa",
        help="输出目录路径（默认: data/narrativeqa）",
    )
    args = parser.parse_args()
    
    process_narrativeqa_dataset(args.output_dir)


if __name__ == "__main__":
    main()

