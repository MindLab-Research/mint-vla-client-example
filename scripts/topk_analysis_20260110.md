# Top-K Analysis at Corrupted Positions

Date: 2026-01-10
Checkpoint: tinker://localhost/vePFS-Mindverse/share/code/tinker-server/checkpoints/d01393b8-7dcb-47b1-a95c-477e81b22498_0/debug_checkpoint

## Key Finding: Off-by-One Alignment Issue

The saved comparison data has an alignment issue:
- Megatron logprobs[i] = P(target_tokens[i] | input_tokens[:i+1]) = P(token[i+1] | prefix)
- vLLM prompt_logprobs[i] = P(input_tokens[i] | input_tokens[:i]) = P(token[i] | prefix-1)

To compare the SAME token probability:
- megatron_trained[i] should compare with vllm_trained[i+1]

## Corrupted Positions Analysis

### Position 5: target='user' (token 2482)
- Megatron trained: -47.75 (catastrophically wrong)
- vLLM trained[6]: -3.79 (slightly improved from fresh -4.05)
- vLLM top-1 at pos 5: '>' with logprob -0.0001 (different context - pos 5 predicts what comes after '<|im_start|')
- vLLM top-1 at pos 6: 'assistant' with logprob -0.043
- 'user' is rank 2 at pos 6 with logprob -3.79

### Position 11: target='10' (token 795)
- Megatron trained: -5.50
- vLLM shows ' ' (space) as top-1 at this position
- Target '10' NOT in vLLM top-10

### Position 14: target='1' (token 16)
- Megatron trained: -29.56
- vLLM shows ' ' (space) as top-1
- Target '1' NOT in vLLM top-10

### Position 21: target='<|im_end|>' (token 163586)
- Megatron trained: -52.23
- vLLM shows '.' as top-1
- Target '<|im_end|>' NOT in vLLM top-10

### Position 23: target='<' (token 27)
- Megatron trained: -47.125
- vLLM shows '\n' as top-1
- Target '<' NOT in vLLM top-10

## Critical Insight

Training made target tokens WORSE in Megatron:
- Fresh logprob for 'user': ~-4.05
- Trained logprob for 'user': -47.75 (CATASTROPHIC DECREASE)

But vLLM with SAME weights shows slight improvement:
- Fresh (approximated): ~-4
- Trained: -3.79 (slight improvement)

## Conclusion

The bug is in Megatron's forward pass after training, NOT in the weights themselves.
With identical trained LoRA weights:
- vLLM produces reasonable results (slight improvement)
- Megatron produces catastrophically wrong results (10^-21 probability)

Possible causes to investigate:
1. LoRA weight application differs between Megatron and vLLM
2. MoE expert routing differs
3. Numerical precision issues in Megatron forward
4. Hidden state corruption at specific layers
