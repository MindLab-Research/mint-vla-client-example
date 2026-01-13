#!/usr/bin/env python3
"""Analyze the catastrophic positions where Megatron diverges from vLLM.

From the log, these positions show Megatron getting 20-40 nats WORSE while vLLM improves:
- pos=10, 13, 17, 21, 23, 25, 30

Goal: Find what these positions have in common (token type, position in sequence, etc.)
"""

from transformers import AutoTokenizer

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

TEST_TEXT = """<|im_start|>user
Count down from 10 to 1, one number per line.<|im_end|>
<|im_start|>assistant
10
9
8
7
6
5
4
3
2
1<|im_end|>"""

# Positions that show catastrophic Megatron degradation
CATASTROPHIC_POSITIONS = [10, 13, 17, 21, 23, 25, 30]

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Input tokens: {len(input_tokens)}")

    print("\n" + "=" * 100)
    print("FULL TOKEN SEQUENCE")
    print("=" * 100)
    print(f"{'Pos':>4} {'InpTok':>8} {'Input Token':20} {'TgtTok':>8} {'Target Token':20}")
    print("-" * 100)

    for i, (inp, tgt) in enumerate(zip(input_tokens, target_tokens)):
        inp_str = tokenizer.decode([inp])
        tgt_str = tokenizer.decode([tgt])
        marker = " *** CATASTROPHIC" if i in CATASTROPHIC_POSITIONS else ""
        print(f"{i:4d} {inp:8d} {repr(inp_str):20} {tgt:8d} {repr(tgt_str):20}{marker}")

    print("\n" + "=" * 100)
    print("CATASTROPHIC POSITIONS ANALYSIS")
    print("=" * 100)

    for pos in CATASTROPHIC_POSITIONS:
        if pos < len(input_tokens):
            inp_tok = input_tokens[pos]
            tgt_tok = target_tokens[pos]
            inp_str = tokenizer.decode([inp_tok])
            tgt_str = tokenizer.decode([tgt_tok])

            # Check for special patterns
            is_newline_input = inp_str == '\n'
            is_space_target = tgt_tok == 220  # Common space token ID
            is_special = '<|' in inp_str or '<|' in tgt_str

            print(f"\nPosition {pos}:")
            print(f"  Input:  token={inp_tok:6d}, text={repr(inp_str)}")
            print(f"  Target: token={tgt_tok:6d}, text={repr(tgt_str)}")
            print(f"  Patterns: newline_input={is_newline_input}, space_target={is_space_target}, special={is_special}")

    # Look for patterns
    print("\n" + "=" * 100)
    print("PATTERN ANALYSIS")
    print("=" * 100)

    # Count target token types at catastrophic positions
    target_tokens_at_catastrophic = [target_tokens[p] for p in CATASTROPHIC_POSITIONS if p < len(target_tokens)]
    print(f"\nTarget tokens at catastrophic positions: {target_tokens_at_catastrophic}")

    # Check if space token (220) is common
    space_count = sum(1 for t in target_tokens_at_catastrophic if t == 220)
    print(f"Space tokens (220): {space_count}/{len(target_tokens_at_catastrophic)}")

    # Check input tokens
    input_tokens_at_catastrophic = [input_tokens[p] for p in CATASTROPHIC_POSITIONS if p < len(input_tokens)]
    print(f"\nInput tokens at catastrophic positions: {input_tokens_at_catastrophic}")

    # Check for newline (198)
    newline_count = sum(1 for t in input_tokens_at_catastrophic if t == 198)
    print(f"Newline tokens (198) in input: {newline_count}/{len(input_tokens_at_catastrophic)}")


if __name__ == "__main__":
    main()
