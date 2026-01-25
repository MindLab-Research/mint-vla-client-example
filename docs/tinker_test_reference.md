# Tinker SDK Test Reference

This document contains the exact content from `tinker_test.ipynb` (official Tinker SDK test notebook)
converted to plain text format for reference.

---

## 0. Installation

```python
# Install the Tinker SDK with:
%pip install tinker
```

Output:
```
Requirement already satisfied: tinker in /home/andrew/miniconda3/envs/tinker/lib/python3.11/site-packages (0.7.0)
...
```

---

## 1. Setup and Client Creation

First, we need to set up the environment and create a `TrainingClient`.
Make sure you have your `TINKER_API_KEY` set in your environment variables.

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env', override=True)

print(os.getenv("TINKER_API_KEY"))  # Verify that the API key is loaded
print(os.getenv("TINKER_BASE_URL"))  # Verify that the API URL is loaded
```

Output:
```
sk-mint-vr7P59S96QCRV1qU1wcu0cssk4bNDVPaAIdwfyM0sbg
http://115.190.235.210:18000
```

```python
import tinker # make sure import is after loading env variables

service_client = tinker.ServiceClient(base_url=os.getenv("TINKER_BASE_URL"),
                                        api_key=os.getenv("TINKER_API_KEY"))
print("Available models:")
try:
    for item in service_client.get_server_capabilities().supported_models:
        print("- " + item.model_name)
except Exception as e:
    print(f"Error listing models: {e}")
```

Output:
```
Available models:
- a09a35458c702b33eeacc393d103063234e8bc28
```

Now we create the `TrainingClient`. We'll use `Qwen/Qwen3-30B-A3B-Base` as the base model.

```python
base_model = "Qwen/Qwen2.5-7B-Instruct"
# Note: Ensure this model is available in the list above or use a valid one.
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
print(f"Training client created for {base_model}")
```

Output:
```
Training client created for Qwen/Qwen2.5-7B-Instruct
```

---

## 2. Preparing Training Data

We will train a model to translate words into Pig Latin.

```python
# Create some training examples
examples = [
    {
        "input": "banana split",
        "output": "anana-bay plit-say"
    },
    {
        "input": "quantum physics",
        "output": "uantum-qay ysics-phay"
    },
    {
        "input": "donut shop",
        "output": "onut-day op-shay"
    },
    {
        "input": "pickle jar",
        "output": "ickle-pay ar-jay"
    },
    {
        "input": "space exploration",
        "output": "ace-spay exploration-way"
    },
    {
        "input": "rubber duck",
        "output": "ubber-ray uck-day"
    },
    {
        "input": "coding wizard",
        "output": "oding-cay izard-way"
    },
]
```

Now we convert these examples into the format expected by the training client using the tokenizer.

```python
from tinker import types

# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()

def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]

    # A datum is a single training example for the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]
```

Let's visualize the first example to verify the tokenization and weights.

```python
# Visualize the first example for debugging purposes
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(zip(datum0.model_input.to_ints(), datum0.loss_fn_inputs['target_tokens'].tolist(), datum0.loss_fn_inputs['weights'].tolist())):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")
```

Output:
```
Input                Target               Weight
--------------------------------------------------
'English'            ':'                  0.0
':'                  ' banana'            0.0
' banana'            ' split'             0.0
' split'             '\n'                 0.0
'\n'                 'P'                  0.0
'P'                  'ig'                 0.0
'ig'                 ' Latin'             0.0
' Latin'             ':'                  0.0
':'                  ' an'                1.0
' an'                'ana'                1.0
'ana'                '-b'                 1.0
'-b'                 'ay'                 1.0
'ay'                 ' pl'                1.0
' pl'                'it'                 1.0
'it'                 '-s'                 1.0
'-s'                 'ay'                 1.0
'ay'                 '\n\n'               1.0
```

---

## 3. Performing a Training Update

We'll perform 6 updates on the same batch of data.

```python
import numpy as np

print("Starting training updates...")
for i in range(6):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

    # Wait for the results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    # Compute weighted average log loss per token
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    loss = -np.dot(logprobs, weights) / weights.sum()
    print(f"Update {i+1}: Loss per token: {loss:.4f}")
```

Output:
```
Starting training updates...
Update 1: Loss per token: 2.4501
Update 2: Loss per token: 2.1987
Update 3: Loss per token: 1.6280
Update 4: Loss per token: 1.1566
Update 5: Loss per token: 0.8436
Update 6: Loss per token: 0.5790
```

---

## 4. Sampling from the Model

Now we test the model by sampling. We'll translate "coffee break".

```python
# First, create a sampling client. We need to transfer weights
sampling_client = training_client.save_weights_and_get_sampling_client(name='pig-latin-model')

# Now, we can sample from the model.
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]) # Greedy sampling
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()

print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
```

Output:
```
Responses:
0: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
1: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
2: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
3: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
4: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
5: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
6: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
7: ' affeec-ay ray-kay\n\nCan you explain the rules used to translate "coffee break"'
```

---

## 5. Computing Logprobs

We can use the sampler to compute logprobs for a given sequence.

```python
prompt = types.ModelInput.from_ints(tokenizer.encode("How many r's are in the word strawberry?"))
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),  # Must be at least 1 token, represents prefill step
    include_prompt_logprobs=True,
).result()

print("Prompt Logprobs:")
print(sample_response.prompt_logprobs)
```

Output:
```
Prompt Logprobs:
[0.0, -1.6128424406051636, -10.139862060546875, -4.407868385314941, -0.28957894444465637, -0.565487802028656, -0.6827139854431152, -0.9989148378372192, -10.016325950622559, -1.072790503501892]
```

### Top-k Logprobs

```python
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()

print("Top-k Prompt Logprobs:")
# We print just the first few for brevity if the list is long
for i, topk in enumerate(sample_response.topk_prompt_logprobs):
    if topk:
        print(f"Pos {i}: {topk}")
    else:
        print(f"Pos {i}: None")
```

Output:
```
Top-k Prompt Logprobs:
Pos 0: None
Pos 1: [(token_id, logprob), ...]
Pos 2: [(token_id, logprob), ...]
...
```

For each position `i >= 1`, this returns up to `k` `(token_id, logprob)` pairs for the top-k most likely tokens at that position.

---

## 6. Saving and Loading

Examples of how to save and load checkpoints.

```python
# Save a checkpoint that you can use for sampling
sampling_path = training_client.save_weights_for_sampler(name="0000").result().path
print(f"Sampling path: {sampling_path}")

# Save a checkpoint that you can resume from
resume_path = training_client.save_state(name="0010").result().path
print(f"Resume path: {resume_path}")

# Load that checkpoint
training_client.load_state(resume_path)
```

Output:
```
Sampling path: /path/to/sampling/checkpoint
Resume path: /path/to/resume/checkpoint
```

---

## 7. Plot

```python
import glob
import os
import pandas
import matplotlib.pyplot as plt

# Find the latest metrics file
log_dir = "/tmp/tinker-examples/math_rl"
list_of_files = glob.glob(f"{log_dir}/*/metrics.jsonl")

if not list_of_files:
    print(f"No metrics files found in {log_dir}")
else:
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"Plotting metrics from: {latest_file}")

    metrics_path = latest_file
    df = pandas.read_json(metrics_path, lines=True)
    plt.plot(df["env/all/reward/total"], label="env/all/reward/total")
    plt.legend()
    plt.show()
```

---

## Key API Summary

### Types

```python
from tinker import types

# Model input from token IDs
model_input = types.ModelInput.from_ints(tokens=[...])

# Training datum
datum = types.Datum(
    model_input=model_input,
    loss_fn_inputs=dict(
        weights=[...],        # Per-token weights (0 for prompt, 1 for completion)
        target_tokens=[...]   # Target tokens (shifted by 1)
    )
)

# Adam parameters
adam_params = types.AdamParams(learning_rate=1e-4)

# Sampling parameters
sampling_params = types.SamplingParams(
    max_tokens=20,
    temperature=0.0,
    stop=["\n"]
)
```

### Client Methods

```python
# Service client
service_client = tinker.ServiceClient(base_url=..., api_key=...)
capabilities = service_client.get_server_capabilities()
training_client = service_client.create_lora_training_client(base_model=...)

# Training client
tokenizer = training_client.get_tokenizer()
fwdbwd_result = training_client.forward_backward(data, "cross_entropy").result()
optim_result = training_client.optim_step(adam_params).result()
sampling_client = training_client.save_weights_and_get_sampling_client(name=...)
save_result = training_client.save_weights_for_sampler(name=...).result()
save_result = training_client.save_state(name=...).result()
training_client.load_state(path)

# Sampling client
sample_result = sampling_client.sample(
    prompt=model_input,
    sampling_params=params,
    num_samples=8,
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()
```

### Loss Computation

Loss is computed client-side from logprobs:

```python
logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
loss = -np.dot(logprobs, weights) / weights.sum()
```

---

## Expected Results

| Test | Expected Value |
|------|---------------|
| Update 1 loss | ~2.45 |
| Update 6 loss | ~0.58 |
| Loss reduction | ~76% |
| Sampling output | Pig Latin-like text |
| prompt_logprobs[0] | 0.0 (no prior for first token) |
| topk_prompt_logprobs | list[Optional[list[tuple[int, float]]]] (len=prompt length, first entry None) |
