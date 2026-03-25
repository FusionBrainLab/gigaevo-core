#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np
from datasets import load_dataset


def to_serializable(obj):
    """将 numpy / 其他非常规类型转换为 JSON 可序列化类型。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def split_input(text: str, dataset: str):
    """
    按数据集类型拆成 (example, instruction, context, question)。
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return "", "", "", ""

    example = ""
    instruction = ""
    context = ""
    question = ""

    # ====== CWE ======
    if dataset == "cwe":
        cwe_instr = (
            "Below is a numbered list of words. In these words, some appear more often than others. "
            "Memorize the ones that appear most often."
        )

        # 找到所有指令出现位置
        instr_positions = []
        start = 0
        while True:
            idx = text.find(cwe_instr, start)
            if idx == -1:
                break
            instr_positions.append(idx)
            start = idx + len(cwe_instr)

        if instr_positions:
            # 使用最后一次出现作为真实 instruction
            instr_start = instr_positions[-1]
            instr_end = instr_start + len(cwe_instr)
            instruction = cwe_instr

            # 之前所有内容当作 example（可能包含完整示例 QA）
            if instr_start > 0:
                example = text[:instr_start].strip()

            # 在该 instruction 之后找第一个 Question:
            q_pos = text.find("Question:", instr_end)
            if q_pos != -1:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                # 没有 Question，剩余全当 context
                context = text[instr_end:].strip()
        else:
            # 没找到固定指令，就尝试简单分割：context + question
            q_pos = text.find("Question:")
            if q_pos != -1:
                context = text[:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text

        return example, instruction, context, question

    # ====== FWE ======
    if dataset == "fwe":
        fwe_instr = (
            "Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words."
        )

        q_pos = text.rfind("Question:")
        if q_pos != -1:
            question = text[q_pos:].strip()
        else:
            q_pos = -1

        instr_start = text.find(fwe_instr)
        if instr_start != -1:
            instr_end = instr_start + len(fwe_instr)
            instruction = fwe_instr
            end_for_context = q_pos if q_pos != -1 else len(text)
            context = text[instr_end:end_for_context].strip()
        else:
            context = text if q_pos == -1 else text[:q_pos].strip()

        return example, instruction, context, question

    # ====== NIAH_MULTIKEY_1 ======
    if dataset == "niah_multikey_1":
        magic_instr = (
            "A special magic number is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the number afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What is the special magic number"
            q_pos = text.rfind(q_key)

            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== NIAH_MULTIKEY_2 ======
    if dataset == "niah_multikey_2":
        magic_instr = (
            "A special magic number is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the number afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What is the special magic number"
            q_pos = text.rfind(q_key)

            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== NIAH_MULTIKEY_3 ======
    if dataset == "niah_multikey_3":
        magic_instr = (
            "A special magic uuid is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the uuid afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What is the special magic uuid"
            q_pos = text.rfind(q_key)

            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== NIAH_MULTIQUERY ======
    if dataset == "niah_multiquery":
        magic_instr = (
            "Some special magic numbers are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the numbers afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What are all the special magic numbers"
            q_pos = text.rfind(q_key)
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== NIAH_MULTIVALUE ======
    if dataset == "niah_multivalue":
        magic_instr = (
            "Some special magic numbers are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the numbers afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What are all the special magic numbers"
            q_pos = text.rfind(q_key)
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== NIAH_SINGLE_1 ======
    if dataset == "niah_single_1":
        magic_instr = (
            "A special magic number is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the number afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What is the special magic number"
            q_pos = text.rfind(q_key)
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== NIAH_SINGLE_2 ======
    if dataset == "niah_single_2":
        magic_instr = (
            "A special magic number is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the number afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What is the special magic number"
            q_pos = text.rfind(q_key)
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== NIAH_SINGLE_3 ======
    if dataset == "niah_single_3":
        magic_instr = (
            "A special magic uuid is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the uuid afterwards."
        )

        instr_start = text.find(magic_instr)
        if instr_start != -1:
            instr_end = instr_start + len(magic_instr)
            instruction = magic_instr

            q_key = "What is the special magic uuid"
            q_pos = text.rfind(q_key)
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()
        else:
            context = text

        return example, instruction, context, question

    # ====== QA_1 / QA_2 ======
    if dataset in ("qa_1", "qa_2"):
        qa_instr = (
            "Answer the question based on the given documents. Only give me the answer and do not output any other words."
        )

        instr_start = text.find(qa_instr)
        if instr_start != -1:
            instr_end = instr_start + len(qa_instr)
            instruction = qa_instr
        else:
            instr_end = 0

        q_pos = text.rfind("Question:")
        if q_pos != -1:
            # question 从最后一次 QA 指令或 Question: 开始
            context = text[instr_end:q_pos].strip()
            question = text[q_pos:].strip()
        else:
            context = text[instr_end:].strip()

        return example, instruction, context, question

    # ====== VT ======
    if dataset == "vt":
        vt_instr = "Memorize and track the chain(s) of variable assignment hidden in the following text."

        first = text.find(vt_instr)
        second = text.find(vt_instr, first + len(vt_instr)) if first != -1 else -1

        if first != -1 and second != -1:
            example = text[first:second].strip()
            instruction = vt_instr

            rest = text[second:].strip()
            instr_start2 = rest.find(vt_instr)
            if instr_start2 != -1:
                instr_end2 = instr_start2 + len(vt_instr)
            else:
                instr_end2 = 0

            q_pos = rest.rfind("Question:")
            if q_pos != -1 and q_pos > instr_end2:
                context = rest[instr_end2:q_pos].strip()
                question = rest[q_pos:].strip()
            else:
                context = rest[instr_end2:].strip()
        else:
            instr_start = text.find(vt_instr)
            if instr_start != -1:
                before = text[:instr_start].strip()
                if before:
                    example = before
                instr_end = instr_start + len(vt_instr)
                instruction = vt_instr
            else:
                instr_end = 0

            q_pos = text.rfind("Question:")
            if q_pos != -1 and q_pos > instr_end:
                context = text[instr_end:q_pos].strip()
                question = text[q_pos:].strip()
            else:
                context = text[instr_end:].strip()

        return example, instruction, context, question

    # ====== 默认 ======
    context = text
    return example, instruction, context, question


def process_dataset(dataset, dataset_name: str, output_path: str):
    """处理单个数据集子集并保存为 jsonl 文件"""
    text_col = None
    for cand in ["input", "text", "content", "prompt"]:
        if cand in dataset.column_names:
            text_col = cand
            break
    if text_col is None:
        raise ValueError(
            f"Could not find text column in {dataset.column_names}, "
            f"please adjust column name in process_dataset."
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cols = dataset.column_names

    with open(output_path, "w", encoding="utf-8") as f_out:
        for row in dataset:
            raw_text = row[text_col]

            example, instruction, context, question = split_input(raw_text, dataset_name)

            sample = {
                "example": example,
                "instruction": instruction,
                "context": context,
                "question": question,
            }

            # 保留所有非文本列，比如 index / outputs
            for k in cols:
                if k == text_col:
                    continue
                v = row[k]
                sample[k] = v

            line = json.dumps(sample, ensure_ascii=False, default=to_serializable)
            f_out.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lighteval/RULER-131072-Qwen2.5-Instruct",
        help="Dataset name to load from HuggingFace",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ruler",
        help="Output directory for jsonl files",
    )
    args = parser.parse_args()

    # 加载数据集
    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)
    print(f"Loaded dataset with splits: {list(ds.keys())}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理每个子集
    for split_name in ds.keys():
        print(f"Processing split: {split_name}")
        output_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        process_dataset(ds[split_name], split_name, output_path)
        print(f"Saved {split_name} to {output_path}")

    print("All splits processed successfully!")


if __name__ == "__main__":
    main()
