"""文本分词、分块训练、评估和自回归生成示例。"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Sequence

# 支持直接运行本文件；作为模块导入时不修改搜索路径。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_torch import (
    SGD, Transformer, Value, _Meta, cross_entropy, get_max_index, no_grad,
)


def make_training_blocks(
    tokens: Sequence[int], block_size: int
) -> list[tuple[Value, list[int]]]:
    """每块读取至多 T+1 个 token，组成错位的输入和标签；保留不足 T 的尾块。"""
    if block_size < 1:
        raise ValueError("block_size 必须大于 0")
    if len(tokens) < 2:
        raise ValueError("训练文本至少需要两个 token")

    blocks = []
    for start in range(0, len(tokens) - 1, block_size):
        block = list(tokens[start : start + block_size + 1])
        inputs = Value(
            _Meta(block[:-1], (len(block) - 1,), dtype="int32"), require_grad=False
        )
        blocks.append((inputs, block[1:]))
    return blocks



def evaluate(
    model: Transformer, blocks: Sequence[tuple[Value, list[int]]]
) -> tuple[float, float]:
    """统计平均交叉熵和 token 正确率，按 token 数加权。"""
    total_loss = 0.0
    correct = 0
    count = 0
    with no_grad():
        for inputs, targets in blocks:
            logits = model(inputs)
            total_loss += cross_entropy(logits, targets).data[0] * len(targets)
            predictions = get_max_index(logits)
            for predicted, target in zip(predictions.data, targets, strict=True):
                if predicted == target:
                    correct += 1
            count += len(targets)
    return total_loss / count, correct / count



def generate(
    model: Transformer,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 32,
    *,
    eos_token_id: int | None = None,
) -> list[int]:
    """自回归解码，采用贪心策略。"""
    if not prompt_ids:
        raise ValueError("提示文本至少需要一个 token")
    tokens = list(prompt_ids)
    with no_grad():
        for _ in range(max_new_tokens):
            context = tokens[-model.max_sequence_len :]
            inputs = Value(
                _Meta(context, (len(context),), dtype="int32"), require_grad=False
            )
            logits = model(inputs)
            last_row = logits.data[-model.vocab_size :]
            next_token = max(range(model.vocab_size), key=last_row.__getitem__)
            tokens.append(next_token)
            if next_token == eos_token_id:
                break
    return tokens



def main() -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description="mini-torch 文本训练与生成")
    parser.add_argument(
        "--text-file",
        type=Path,
        default=Path(__file__).parent / "data" / "train_text.txt",
        help="UTF-8 训练文本路径",
    )
    parser.add_argument(
        "--tokenizer",
        default="hf-internal-testing/tiny-random-gpt2",
        help="Hugging Face 仓库名或本地 tokenizer.json 路径",
    )
    parser.add_argument("--prompt", default="the cat", help="生成时的提示文本")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.is_file():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_pretrained(args.tokenizer)
    # 文本由本框架分块，保留原始 token，不让分词器预先截断或补齐。
    tokenizer.no_truncation()
    tokenizer.no_padding()

    text = args.text_file.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is not None:
        tokens.append(eos_token_id)
    blocks = make_training_blocks(tokens, args.block_size)

    model = Transformer(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=16,
        hidden_dim=32,
        attention_bias=True,
        fnn_bias1=True,
        fnn_bias2=True,
        lm_bias=True,
        max_sequence_len=args.block_size,
        dtype=args.dtype,
        rng=random.Random(22),
    )
    optimizer = SGD(model.parameters(), lr=args.lr)
    print(f"词表大小：{model.vocab_size}；语料 token长度：{len(tokens)}；训练块大小：{len(blocks)}")
    initial_loss, _ = evaluate(model, blocks)
    print(f"训练前 loss：{initial_loss:.4f}")

    for step in range(args.steps):
        inputs, targets = blocks[step % len(blocks)]
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"step {step}: loss={loss.data[0]:.4f}")

    final_loss, accuracy = evaluate(model, blocks)
    print(f"训练后 loss：{final_loss:.4f}；训练集正确率：{accuracy:.2%}")
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    output_ids = generate(
        model, prompt_ids, args.max_new_tokens, eos_token_id=eos_token_id
    )
    print("提示文本：", args.prompt)
    print("生成文本：", tokenizer.decode(output_ids, skip_special_tokens=True))



if __name__ == "__main__":
    main()
