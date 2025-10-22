# nanochat Architecture Documentation

## Overview

nanochat is a full-stack implementation of a ChatGPT-like LLM, designed to run on a single 8xH100 GPU node within a $100-300 budget. It implements a complete pipeline from tokenization through pretraining, continuous training, supervised finetuning (SFT), and reinforcement learning (RL). The entire project is ~8,300 lines of code across 45 files, making it highly readable and hackable.

### Key Design Philosophy
- **Minimalist**: Single, cohesive codebase with no excessive configuration
- **Learnable**: Clean implementations of standard LLM components
- **Hackable**: Full transparency from data loading to inference
- **End-to-end**: Complete pipeline from raw data to a web UI chat interface

---

## Architecture Overview

The codebase is organized into three main directories:

**nanochat/**: Core library modules for model, training, and inference
- gpt.py: Transformer model implementation
- engine.py: Inference engine with KV caching
- tokenizer.py: BPE tokenizer using rustbpe + tiktoken
- dataloader.py: Distributed data loading
- dataset.py: Parquet dataset management
- checkpoint_manager.py: Model saving/loading utilities
- loss_eval.py, core_eval.py: Evaluation metrics
- adamw.py, muon.py: Optimizer implementations
- common.py: Utilities (logging, distributed setup)

**scripts/**: Training and evaluation entry points
- base_train.py: Pretraining on fineweb-edu
- mid_train.py: Continuous training on task mixture
- chat_sft.py: Supervised finetuning for chat
- chat_rl.py: Reinforcement learning with GRPO
- base_eval.py, chat_eval.py: Evaluation scripts
- chat_cli.py, chat_web.py: Inference interfaces
- tok_train.py, tok_eval.py: Tokenizer training/evaluation

**tasks/**: Evaluation and training task definitions
- common.py: TaskMixture abstraction
- gsm8k.py, arc.py, mmlu.py, humaneval.py, smoltalk.py: Task implementations

**rustbpe/**: Rust-based BPE tokenizer for fast training

---

## Core Components

### 1. Model Architecture (gpt.py)

Modern Transformer with key design choices for efficiency and performance:

#### Architectural Features
- **Rotary Position Embeddings (RoPE)**: No explicit position embeddings, uses rotation matrices for relative positions
- **QK Normalization**: Normalizes query/key before attention for stability
- **Multi-Query Attention (MQA)**: Reduces KV cache size by sharing values across heads (configurable via n_kv_head)
- **ReLU² Activation**: In MLP layers instead of GELU
- **Functional RMSNorm**: No learnable parameters in normalization
- **Untied Embeddings**: Separate wte (token embedding) and lm_head weights
- **No Bias**: Linear layers have no bias terms

#### GPTConfig Dataclass
- sequence_len: Context window (default 1024)
- vocab_size: Vocabulary size (default 50304)
- n_layer: Number of transformer blocks (12-32 typical)
- n_head: Number of query heads
- n_kv_head: Number of KV heads (MQA ratio = n_head / n_kv_head)
- n_embd: Embedding dimension

#### Model Scaling Pattern
Base training uses aspect ratio 64: `n_embd = depth * 64`
Example: depth=20 → n_embd=1280, num_heads=10 (ceil_div(1280, 128))

#### Dual Optimizer Strategy
Setup two separate optimizers with different learning rates:
1. **AdamW**: For embeddings (wte) and lm_head (unembedding)
   - Higher learning rates (embedding_lr=0.2, unembedding_lr=0.004)
   - Scaled by sqrt inverse of model dimension
   - Includes weight decay

2. **Muon**: For all matrix parameters (attention, MLP)
   - Different hyperparameters (matrix_lr=0.02)
   - More stable optimization trajectory
   - Momentum scheduled from 0.85 to 0.95

---

## Training Pipeline (4 Stages)

### Stage 1: Base Training (base_train.py)
**Purpose**: Learn language modeling on 38 billion tokens of text

**Dataset**: fineweb-edu-100b-shuffle (parquet files, ~250M chars each)
**Duration**: ~36 hours on 8xH100
**Model Size**: 561M parameters (d20: depth=20, dim=1280)
**Tokens**: 11.2B (following Chinchilla: 20x parameters)

Key Mechanisms:
- Parquet files streamed via parquets_iter_batched() with distributed stride
- On-the-fly tokenization using multi-threaded rustbpe
- Gradient accumulation maintains total batch size of 524K tokens
- Learning rate schedule: No warmup, 20% cooldown at end
- Loss computation: Standard cross-entropy with softcap on logits

### Stage 2: Mid Training (mid_train.py)
**Purpose**: Teach chat patterns and tool use through continuous training

**Data**: TaskMixture of three datasets
- SmolTalk (460K examples): General conversation
- MMLU (100K examples): Multiple choice knowledge
- GSM8K (8K examples): Math with calculator tool use

**Duration**: ~8 hours
**Load From**: Base checkpoint

Key Differences from Base Training:
- Data comes from task abstractions, not parquet files
- Conversations tokenized via render_conversation() which returns (ids, mask)
- Progress-based learning rate schedule (not iteration-based)
- Uses task.reward() for RL-style evaluation

### Stage 3: SFT (Supervised Finetuning) (chat_sft.py)
**Purpose**: Fine-tune on high-quality chat examples with selective training

**Data**: TaskMixture of 21.4K examples
- ARC-Easy (2.3K), ARC-Challenge (1.1K)
- GSM8K (8K)
- SmolTalk (10K, subset)

**Duration**: ~2 hours
**Load From**: Mid checkpoint

Unique Mechanism - Masked Loss:
- render_conversation() returns (ids, mask) for each example
- Pad variable-length conversations in batch
- Loss computed only on mask=1 tokens (assistant responses)
- Tokens with mask=0 (user input, tool use) are masked with ignore_index=-1

### Stage 4: Reinforcement Learning (chat_rl.py)
**Purpose**: Optimize GSM8K performance using GRPO algorithm

**Algorithm**: Simplified GRPO (on-policy, no KL penalty, no PPO ratio)
1. Sample K=16 completions per example using engine.generate_batch()
2. Evaluate with task.reward() (0/1 for correct/incorrect)
3. Compute token-level advantages: reward - mean_reward
4. Weight loss by advantage (GAPO normalization)
5. Mask non-sampled tokens (prompt, tool use) in loss

**Duration**: ~1 hour (optional stage)
**Data**: GSM8K training set (8K examples)

---

## Tokenization (tokenizer.py + rustbpe/)

#### Hybrid Architecture
- **rustbpe** (Rust): Fast BPE training on raw text, outputs mergeable ranks
- **tiktoken** (Python): Inference encoding using precomputed ranks
- **Special tokens**: 9 fixed tokens for chat structure and tool use

#### Special Tokens (SPECIAL_TOKENS)
<|bos|>: Beginning of sequence (prepended to every document)
<|user_start|>, <|user_end|>: User message delimiters
<|assistant_start|>, <|assistant_end|>: Assistant message delimiters
<|python_start|>, <|python_end|>: Python tool delimiters
<|output_start|>, <|output_end|>: Tool output delimiters

#### render_conversation() Method
Converts conversation dicts to tokenized form with training masks:
- Input: dict with "messages" list (each message has "role" and "content")
- Output: (ids, mask) tuple
- Mask logic:
  - mask=0 for BOS token
  - mask=0 for user messages and their delimiters
  - mask=0 for tool use tokens (python_start/end, output_start/end)
  - mask=1 for assistant response tokens

---

## Inference Engine (engine.py)

#### KVCache Class
Manages key-value cache for efficient incremental decoding:
- insert_kv(layer_idx, k, v): Insert new KV for layer, return full view
  - Dynamically grows cache in 1024-token increments
  - Auto-increments self.pos after last layer processes
- prefill(other_cache): Clone another cache (used for batch expansion)

#### Engine Class
Efficient generation with multiple sampling modes and tool use.

**Streaming Generation**:
```python
for token_column, token_masks in engine.generate(
    tokens, num_samples=4, max_tokens=256,
    temperature=1.0, top_k=50, seed=42
):
    # token_column: [next_token for each sample]
    # token_masks: [0=forced, 1=sampled for each sample]
```

**Batch Generation** (returns full sequences):
```python
sequences, masks = engine.generate_batch(
    tokens, num_samples=4, max_tokens=256, ...
)
```

#### Tool Use State Machine
Engine handles Python calculator tool automatically:
1. When <|python_start|> sampled: Enter python block, accumulate tokens
2. When <|python_end|> sampled: Exit block, try to evaluate expression
3. If evaluation succeeds:
   - Force injection of <|output_start|> + result + <|output_end|>
   - These forced tokens have mask=0 (won't be trained on them in RL)

---

## Data Loading Infrastructure

#### Base Training: parquet_iter_batched() (dataloader.py)
Streams parquet files with distributed stride:
- Each rank accesses documents at indices: rank, rank+world_size, rank+2*world_size
- Ensures unique data across ranks without shuffling
- Works with DDP synchronization

#### Mid/Chat Training: TaskMixture (tasks/common.py)
Unified interface for multiple datasets:
```python
dataset = TaskMixture([
    SmolTalk(split="train"),
    MMLU(subset="auxiliary_train", split="train"),
    GSM8K(subset="main", split="train"),
])
```

#### RL Training: Direct Task Access
```python
task = GSM8K(subset="main", split="train")
for example_idx in range(len(task)):
    conversation = task[example_idx]
    reward = task.reward(conversation, generated_text)
```

---

## Training Mechanics

### Optimizer Strategy
**Two-Optimizer Design**:
1. **AdamW** for embeddings + lm_head
   - Embedding LR: 0.2 (higher, embedding is large)
   - Unembedding LR: 0.004 (lower, output projection)
   - Weight decay: 0.0 (no decay on embeds)
   - LR scaled by (model_dim / 768)^-0.5 for size-invariant tuning

2. **Muon** for all matrix parameters
   - Matrix LR: 0.02
   - Momentum: starts at 0.85, ramps to 0.95
   - More stable than Adam for weight matrices

### Batch Size and Gradient Accumulation
```
tokens_per_fwdbwd = device_batch_size * seq_len * world_size
grad_accum_steps = total_batch_size / tokens_per_fwdbwd
```

Memory-Compute Tradeoff:
- Reduce device_batch_size to fit GPU memory
- Grad accumulation auto-scales to maintain effective batch size
- Example: 524K target, 8 GPUs, seq_len=2048, device_bs=32
  - Tokens/fwdbwd = 32 * 2048 * 8 = 524,288
  - grad_accum_steps = 1 (perfect fit!)

---

## Full Pipeline Execution (speedrun.sh)

Complete training run in ~4 hours on 8xH100:

Phase 1: Environment Setup (~5 min)
- Install uv package manager, create venv, install dependencies

Phase 2: Tokenizer Training (~30 min)
- Compile rustbpe, download data shards, train BPE on 2B chars

Phase 3: Base Training (~1.5 hours)
- Train d20 model on 11.2B tokens from fineweb-edu

Phase 4: Mid Training (~50 min)
- Train on SmolTalk + MMLU + GSM8K mixture

Phase 5: SFT (~30 min)
- Supervised finetune on ARC + GSM8K + SmolTalk with masked loss

Phase 6: Evaluation & Reporting (~5 min)
- Generate report.md with system info and metrics table

Total Time: ~4 hours
Total Cost: ~$100 (33 hours at $3/GPU/hour on H100)

---

## Key Design Decisions

### Why Muon + AdamW instead of single optimizer?
- Muon is newer (2024), shows better stability for weight matrices
- AdamW remains useful for embeddings (smaller, benefit from decay)
- Different LRs per group empirically outperforms single LR
- Allows independent tuning of embedding vs. matrix learning

### Why RoPE + QK Norm?
- RoPE: Relative positional encoding, extrapolates beyond training length
- QK Norm: Stabilizes attention without learnable parameters
- Functional RMSNorm (no parameters) saves memory vs. LayerNorm

### Why Multi-Query Attention (MQA)?
- Reduces KV cache size from O(B*T*H*D) to O(B*T*1*D) for inference
- Critical for long sequences and batch generation on limited memory
- Configurable: n_kv_head can be 1, 2, or n_head (full MHA)

### Why Hybrid rustbpe + tiktoken?
- rustbpe: Fast BPE training in Rust, parallelizable
- tiktoken: Production-grade inference, highly optimized
- Clean separation: train outputs mergeable ranks, tiktoken consumes them

### Why render_conversation() with masks?
- Flexible token selection: Train on any subset of tokens per example
- Reusable in SFT: Mask user input tokens, train only on assistant
- Reusable in RL: Mask forced tokens (prompt + tool output)
- Simple interface: Returns (ids, mask) tuple

### Why TaskMixture abstraction?
- Unified interface: All tasks support same operations (index, reward, evaluate)
- Easy mixing: Different datasets in single training pass
- Extensible: Add new tasks by implementing interface
- RL-ready: Each task has reward() method for on-policy sampling

---

## Integration Points

### Model ↔ Engine
```python
model, tokenizer, meta = load_model("mid", device, phase="eval")
engine = Engine(model, tokenizer)
sequences, masks = engine.generate_batch(prompt_ids, num_samples=16)
```

### Engine ↔ Tokenizer
- tokenizer.encode(text): Convert text to token IDs
- tokenizer.decode(ids): Convert tokens back to text
- tokenizer.render_conversation(doc): Chat format to (ids, mask)
- tokenizer.encode_special("<|bos|>"): Special token lookup
- tokenizer.get_bos_token_id(): Beginning of sequence token

---

## Evaluation Metrics

### Base Training
- **BPB (Bits Per Byte)**: Cross-entropy loss, computed on validation set
- **CORE**: Aggregate metric from 5 reasoning tasks

### Mid Training
- Same CORE tasks plus HumanEval and ChatCORE

### Chat Training (SFT/RL)
- **Pass@K**: Percentage of problems solved in top-K samples
- Task-specific metrics from underlying benchmarks

---

## Quick Reference Commands

### Full Training Pipeline
```bash
bash speedrun.sh
```

### Base Model Only
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

### Larger Model (d26)
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
```

### Interactive Chat
```bash
python -m scripts.chat_cli -p "Your prompt here"
python -m scripts.chat_web  # Opens http://localhost:8000
```

### Evaluation Only
```bash
torchrun --nproc_per_node=8 -m scripts.base_eval
```

### Parameter Override
All scripts support --flag=value format:
```bash
python -m scripts.base_train -- --depth=26 --matrix_lr=0.015
```

---

## File Organization Summary

| File | Purpose | Key Components |
|------|---------|-----------------|
| gpt.py | Transformer model | GPT, GPTConfig, CausalSelfAttention |
| engine.py | Inference engine | Engine, KVCache, RowState |
| tokenizer.py | BPE tokenization | RustBPETokenizer, render_conversation |
| dataloader.py | Training data | tokenizing_distributed_data_loader |
| dataset.py | Parquet data | parquets_iter_batched, download_single_file |
| checkpoint_manager.py | Model I/O | save_checkpoint, load_checkpoint, build_model |
| adamw.py | AdamW optimizer | DistAdamW |
| muon.py | Muon optimizer | Muon, DistMuon |
| common.py | Utilities | compute_init, print0, get_base_dir |
| loss_eval.py | Validation loss | evaluate_bpb |
| core_eval.py | CORE metric | evaluate_core |

---

## Extension Points

### Adding a New Training Stage
1. Create new script in scripts/ following pattern of base_train.py
2. Define dataset: either parquet via dataloader or Task via TaskMixture
3. Load checkpoint: load_model(source="prev_stage", ...)
4. Save checkpoint: save_checkpoint(checkpoint_dir, step, ...)
5. Add execution to speedrun.sh pipeline
6. Implement evaluation function for metrics

### Adding a New Task
1. Create tasks/my_task.py implementing task interface
2. Required methods:
   - __len__(): Number of examples
   - __getitem__(idx): Returns conversation dict
   - reward(conversation, text): 0/1 reward for RL
   - evaluate(model, engine, ...): Compute metrics
3. Add to TaskMixture in training script
4. Each example must have conversation format with "messages"

### Modifying Inference
1. Extend Engine.generate() or Engine.generate_batch()
2. Update RowState to track additional generation state
3. Use special tokens and forced token injection for control
4. Ensure mask=0 for non-trained tokens

---

## References

- Rotary Embeddings: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- QK Norm: Goyal et al. "Improving Transformer with Depth-wise Adaptive Embedding" (2023)
- MQA: Shazeer et al. "Fast Transformer Decoding: One Write-Head is All You Need" (2019)
- Muon: Zöhrer et al. "The Muon Optimizer" (2024)
- BPE: Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units" (2016)
- Chinchilla Scaling: Hoffmann et al. "Training Compute-Optimal Large Language Models" (2022)
