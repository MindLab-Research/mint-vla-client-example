#!/usr/bin/env python3
"""Identify which positions are in the assistant response vs user message."""

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


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)

    print("Full token sequence:")
    print("=" * 70)

    # Find key positions
    assistant_start = None
    im_end_positions = []

    for i, tok in enumerate(tokens):
        tok_str = tokenizer.decode([tok])

        # Mark special positions
        marker = ""
        if tok_str == "assistant":
            assistant_start = i
            marker = " <-- ASSISTANT START"
        elif tok == 163586:  # <|im_end|>
            im_end_positions.append(i)
            marker = " <-- IM_END"
        elif i > 0 and tokens[i-1] == 163586:
            marker = " <-- after IM_END"

        print(f"  pos={i:2d}: token={tok:6d} ({repr(tok_str):15s}){marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)

    if assistant_start:
        print(f"Assistant keyword at position: {assistant_start}")
        print(f"First assistant CONTENT token: position {assistant_start + 2}")  # +1 for \n
        print(f"  (token {tokens[assistant_start + 2]} = '{tokenizer.decode([tokens[assistant_start + 2]])}')")

    print(f"\nFor training on assistant response:")
    print(f"  Positions to train: {assistant_start + 2} onwards")
    print(f"  Position 7 target 'Count' is in: USER message (should NOT be trained)")


if __name__ == "__main__":
    main()
