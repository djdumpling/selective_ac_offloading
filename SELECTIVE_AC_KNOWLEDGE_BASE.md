# Selective Activation Checkpointing & Offloading: Complete Research Context

> **Purpose**: This document serves as the complete knowledge base for a research project on
> learned selective activation checkpointing and offloading for transformer training. It
> consolidates mathematical foundations, prior work, algorithmic proposals, and practical
> implementation details.

---

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Prior Work: Algorithms & Systems](#2-prior-work-algorithms--systems)
3. [The Frontier: Compression, Offloading, PyTorch APIs](#3-the-frontier-compression-offloading-pytorch-apis)
4. [Algorithmic Proposals (Novel Contributions)](#4-algorithmic-proposals-novel-contributions)
5. [SDPO + TTT-Discover Setup](#5-sdpo--ttt-discover-setup)
6. [Practical Implementation Plan](#6-practical-implementation-plan)

---

## 1. Mathematical Foundations

### 1.1 Why √L evenly-spaced checkpoints cost 33% compute overhead

**Source**: Chen, Xu, Zhang, Guestrin, "Training Deep Nets with Sublinear Memory Cost," arXiv 1604.06174, 2016.

Consider an L-layer feedforward network. Without checkpointing, all L activations must be
stored for backward: O(L) memory. Chen et al. partition the network into √L segments of
√L layers each, saving only the segment-boundary activations (√L checkpoints).

**Memory analysis**: You store √L boundary activations permanently, plus at most √L
intermediate activations during any segment's recomputation = O(2√L) = O(√L) total.

**Compute analysis**: During backpropagation, when you reach a segment boundary, you must
re-run the forward pass for that segment (√L layers) to recompute the intermediates.
There are √L segments, each requiring √L forward operations = √L × √L = L total
recomputed forward operations. But this equals exactly one full forward pass. Since a
training step normally consists of one forward + one backward (backward ≈ 2× forward
in FLOPs), adding one extra forward pass increases total compute from 3F to 4F, which
is 4/3 - 1 = **33% overhead**.

More precisely:
```
Without checkpointing:  1 forward + 1 backward = 3F total  (backward ≈ 2F)
With √L checkpointing:  1 forward + 1 backward + 1 recomputed forward = 4F total
Overhead = (4F - 3F) / 3F = 33%
```

The 33% is an upper bound assuming uniform per-layer cost and complete segment
recomputation. In practice, selective strategies achieve much less overhead by
recomputing only cheap operations.

**Recursive extension**: Store k=1 checkpoint per recursion level, achieving O(log L)
memory at O(L log L) compute. Rarely practical due to large constant factors.

### 1.2 Activation memory per transformer layer: the full derivation

**Source**: Korthikanti, Casper, Lym, McAfee, Patabandige, Shoeybi, Catanzaro, "Reducing
Activation Recomputation in Large Transformer Models," arXiv 2205.05198, MLSys 2023.

**Notation**: s = sequence length, b = micro-batch size, h = hidden dimension, a = number
of attention heads, t = tensor parallelism degree. All sizes in bytes assuming mixed
precision (bf16 for activations = 2 bytes per element, fp32 masks = 1 byte per element).

#### 1.2.1 Attention block activations: 11sbh + 5as²b bytes

Each tensor stored for backpropagation, in order of computation:

| Tensor | Shape | Bytes | Why stored |
|--------|-------|-------|------------|
| Input to QKV projection (LN output) | [s, b, h] | 2sbh | Needed for QKV weight gradient |
| Q tensor | [s, b, h] | 2sbh | Needed for attention score backward |
| K tensor | [s, b, h] | 2sbh | Needed for attention score backward |
| Softmax output (attention weights) | [b, a, s, s] | 2as²b | Needed for V-weighted sum backward |
| Attention dropout mask | [b, a, s, s] | as²b | 1 byte per element (boolean mask) |
| Attention dropout output | [b, a, s, s] | 2as²b | Needed for V projection backward |
| V tensor | (accounted in QKV) | — | — |
| Linear projection input (attn output) | [s, b, h] | 2sbh | Needed for output projection weight grad |
| Output dropout mask | [s, b, h] | sbh | 1 byte per element |

**Subtotal**: (2 + 2 + 2 + 2 + 1)sbh + (2 + 1 + 2)as²b = **11sbh + 5as²b**

Note: V is part of the QKV projection and its memory is counted in the Q, K accounting
above (the QKV projection produces Q, K, V together).

#### 1.2.2 Why the quadratic term 5as²b arises

The quadratic term comes from the attention mechanism's core operation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

The matrix QK^T has shape [s, s] per head per batch element, producing a [b, a, s, s]
tensor. Three derived tensors must be stored:

1. **Softmax output** (2as²b): The post-softmax attention weights are needed to compute
   gradients w.r.t. V (d_loss/dV = Attn_weights^T · d_loss/d_output) and w.r.t. the
   pre-softmax scores (via the softmax Jacobian). Stored in fp16 = 2 bytes/element.

2. **Dropout mask** (as²b): Standard attention dropout applies a mask to the softmax
   output. The mask is boolean (1 byte/element) and must be stored to reproduce the
   same dropped positions during backward.

3. **Dropout output** (2as²b): The post-dropout attention weights are the actual values
   used to weight V. Storing this avoids recomputing dropout(softmax(QK^T/√d_k)) during
   backward. Stored in fp16 = 2 bytes/element.

Total quadratic: (2 + 1 + 2) × a × s² × b = **5as²b bytes**.

**Scale example for GPT-3 175B** (a=96, s=2048, h=12288):
- Linear term: 11sbh = 11 × 2048 × 1 × 12288 × 2 bytes ≈ 555 MB per layer
- Quadratic term: 5as²b = 5 × 96 × 2048² × 1 × 2 bytes ≈ 4.03 GB per layer
- Ratio: quadratic/linear = 5as/h = 5 × 96 × 2048 / 12288 = **80**

The quadratic term dominates by 80× for GPT-3. This is why eliminating it matters so much.

#### 1.2.3 MLP block activations: 19sbh bytes

| Tensor | Shape | Bytes | Why stored |
|--------|-------|-------|------------|
| Input to first linear (LN output) | [s, b, h] | 2sbh | Weight gradient |
| GeLU input (first linear output) | [s, b, 4h] | 8sbh | GeLU backward |
| Second linear input (GeLU output) | [s, b, 4h] | 8sbh | Weight gradient |
| Output dropout mask | [s, b, h] | sbh | Boolean mask |

**Subtotal**: (2 + 8 + 8 + 1)sbh = **19sbh bytes**

#### 1.2.4 LayerNorm activations: 4sbh bytes

Two LayerNorms per transformer layer (one before attention, one before MLP). Each stores
its input for the affine transform gradient: 2 × 2sbh = **4sbh bytes**.

#### 1.2.5 Total per-layer formula

```
Total = (11 + 19 + 4)sbh + 5as²b = sbh(34 + 5as/h)
```

With tensor parallelism degree t and sequence parallelism:
```
Total = sbh(34 + 5as/h) / t
```

#### 1.2.6 How selective recomputation eliminates the quadratic term

**Key insight**: The operations producing the 5as²b tensors — QK^T multiplication,
softmax, and attention dropout — are extremely cheap in FLOPs relative to the memory
they consume. The QK^T matmul is O(as²d_k) = O(as²h/a) = O(s²h) FLOPs, while the
linear projections (QKV and output) are O(sh²) FLOPs. Since s << h for typical configs,
the attention score computation is much cheaper than the projections.

**Selective recomputation strategy**: During forward, discard all three quadratic tensors
(softmax output, dropout mask, dropout output). During backward, recompute them from
Q, K, V (which are kept in memory — they're part of the linear 11sbh budget).

**Memory after selective recomputation**: Remove 5as²b, leaving only:
```
Activation memory = 34 · sbh / t  (per layer)
```

**Compute overhead**: The recomputed operations (QK^T, softmax, dropout) have a
hardware-to-model FLOPS ratio of approximately 1 + s/(6h):
- GPT-3 175B (s=2048, h=12288): overhead = 1 + 2048/73728 = **2.7%**
- MT-NLG 530B (s=2048, h=20480): overhead = **1.6%**

This is dramatically better than the 33% from full recomputation, because we're only
recomputing the cheapest operations that happen to consume the most memory.

#### 1.2.7 How FlashAttention changes the picture

FlashAttention (Dao et al., 2022) never materializes the full s×s attention matrix. It
computes attention in tiles, streaming through Q, K, V blocks and accumulating the
output via online softmax. This means:

- The 5as²b quadratic memory **never exists** when FA is active
- FA's backward pass re-reads Q, K, V from HBM and recomputes attention blocks on-the-fly
- The activation memory for the attention block with FA is approximately:
  ```
  FA attention memory ≈ 2sbh (Q) + 2sbh (K) + 2sbh (V) + 2sbh (output proj input) + sbh (dropout mask)
                      = 9sbh  (no quadratic term)
  ```

**Implication**: With FlashAttention active, the Korthikanti selective strategy
(recompute attention, keep MLP) is no longer optimal. The attention block is already
memory-efficient. The new bottleneck is the **MLP block** (19sbh), specifically:
- GeLU input: 8sbh — expensive matmul to recompute, but potentially compressible
- Second linear input: 8sbh — expensive matmul to recompute, potentially offloadable

This shift means the optimal selective AC strategy should focus on the MLP intermediates,
not the attention intermediates. No existing system has formalized this observation.

### 1.3 Memory model for pipeline parallelism

Under 1F1B (one forward one backward) pipeline scheduling with PP stages and B
micro-batches, a stage at position s from the pipeline's end must stash activations for
(s-1) in-flight micro-batches:

```
Peak_memory(stage, position_s) =
    parameters × (2 + optimizer_multiplier(zero_level)) / TP
    + current_forward_activations(AC_mode) / TP
    + (s - 1) × stashed_activations_per_microbatch(AC_mode) / TP
```

Where:
- optimizer_multiplier: 12 bytes/param for Adam (4 fp32 copies: param, grad, m, v) with ZeRO-0,
  divided by DP degree for ZeRO-1/2/3
- current_forward_activations: whatever the AC strategy retains for the current microbatch
- stashed_activations: boundary inputs retained for later backward passes
- The (s-1) multiplier means early pipeline stages pay a much higher memory cost per
  stashed activation than later stages

---

## 2. Prior Work: Algorithms & Systems

### 2.1 Checkmate (Jain et al., MLSys 2020)

**Core idea**: Formalize activation checkpointing as a Mixed-Integer Linear Program (MILP).

**Formulation**: Two binary decision matrices over n nodes and T timesteps:
- S[t, v] ∈ {0,1}: is node v's output stored in memory at time t?
- R[t, v] ∈ {0,1}: is node v being computed at time t?

**Objective**: min Σ_{t,v} c_v · R[t,v] (minimize total compute cost)

**Constraints**:
1. Memory budget: Σ_v m_v · S[t,v] ≤ M_budget ∀t
2. Dependencies: R[t,v]=1 → S[t,u]=1 for all predecessors u of v
3. Persistence: S[t,v]=1 → (S[t-1,v]=1 OR R[t,v]=1)
4. Correctness: every backward node computed exactly once

**Complexity**: O(n²) binary variables → NP-hard, but solvable by off-the-shelf MILP
solvers (Gurobi) in under an hour for moderate networks.

**Results**: Up to 5.1× larger batch sizes with minimal compute overhead. Dominates all
heuristics (√n, greedy) across VGG, MobileNet, U-Net.

**Limitations**: Doesn't scale past ~500 computation graph nodes. TensorFlow-only
implementation. No offloading support. Requires Gurobi license.

### 2.2 MONeT (Shah et al., ICLR 2021)

**Key extension over Checkmate**: Jointly optimize checkpointing schedule AND operator
implementation choices. Different convolution algorithms (e.g., im2col vs. Winograd),
in-place operations, and ReLU sign storage have different memory-compute tradeoff
profiles. Co-optimizing yields better Pareto frontiers.

**Results**: 3× overall memory reduction at 9-16% compute overhead. 1.2-1.8× less memory
than Checkmate at equal compute cost.

**Implementation**: PyTorch, requires Gurobi. Available at github.com/utsaslab/MONeT.

### 2.3 Beaumont et al. (NeurIPS 2021)

**Key contribution**: First optimal algorithm combining rematerialization AND offloading.

**Insight**: Remat consumes GPU compute bandwidth; offloading consumes PCIe bandwidth.
These are independent resources, so combining them achieves strictly better memory
reduction than either alone.

**Algorithm (POFO)**: Dynamic programming over chain-structured networks. For each layer i,
decide: (a) which forward operator variant to use (cheap vs. memory-efficient),
(b) whether to keep, offload, or discard the activation, and (c) offload timing.

**Complexity**: Polynomial time for linear chains. They prove the general DAG version is
strongly NP-hard.

**Results**: 4-6× activation memory reduction with under 20% overhead. Offloading removes
one-third of recomputation's overhead.

**Limitation**: Only handles linear chains (no skip connections/residuals). Not applicable
to transformers without approximation.

### 2.4 DTR — Dynamic Tensor Rematerialization (Kirisame et al., ICLR 2021)

**Approach**: Runtime (online) eviction decisions using a greedy heuristic. No static
pre-computation of schedules.

**Eviction score**: For each tensor t, compute:
```
score(t) = cost_to_recompute(t) / (memory_freed(t) × staleness(t))
```
Evict the tensor with the lowest score (cheapest to recompute per byte freed).

**Advantages**: Handles dynamic computation graphs (impossible for static ILP/DP). Provably
matches Chen et al.'s O(√n) bound for linear chains.

**Limitation**: No optimality guarantees for general graphs. Runtime overhead from eviction
decisions. Cannot plan ahead (purely reactive).

### 2.5 MOCCASIN (Bartan et al., ICML 2023)

**Key improvement**: Reformulates Checkmate's MILP using constraint programming with
"retention intervals," reducing variable count from O(n²) to O(n).

**Formulation**: For each node v, define a set of intervals [start_i, end_i] during which
v's output is retained in memory. A node can be recomputed at most C_v times
(typically C_v = 2). Uses Google OR-Tools' CP-SAT solver.

**Results**: 10× faster than Checkmate with equivalent solution quality. Scales to graphs
with 1000+ nodes where Checkmate fails.

### 2.6 IBM/Meta Selective AC for FSDP (PyTorch blog, March 2024)

**Approach**: Within PyTorch's FSDP, apply activation checkpointing selectively — e.g.,
checkpoint every other transformer layer, or checkpoint only attention within each
layer. Uses DeepSpeed's `checkpointing` module with modified `_gradient_checkpointing_func`.

**Results**: 10% throughput boost over uniform full checkpointing on Llama-2 7B/70B.
57% MFU on 128 A100s. Open-sourced via `foundation-model-stack/fms-fsdp`.

### 2.7 Korthikanti et al. Selective Recomputation (MLSys 2023)

**The baseline to beat in practice.** Implemented in Megatron-LM as
`--recompute-granularity selective`. Hard-coded to recompute only the attention
core (QK^T, softmax, dropout) while keeping all MLP activations.

**Results on 1T-parameter model**: 29-32% throughput improvement over full recomputation.
56% model FLOPs utilization. Only 1.6-2.7% recomputation overhead.

**Limitation**: Not adaptive — uses the same strategy regardless of model architecture,
hardware, or memory pressure. Does not consider offloading or compression.

### 2.8 PyTorch SAC Memory Budget API (PyTorch 2.4+)

**Mechanism**: `activation_memory_budget = 0.5` with `torch.compile`. Internally:
1. Traces the joint forward-backward graph
2. Runs a min-cut/max-flow partitioner to classify ops as MUST_SAVE vs. PREFER_RECOMPUTE
3. Applies a DP knapsack solver to find the Pareto-optimal subset of ops to save

**Results**: Budget=0.5 achieves ~50% activation memory reduction by recomputing pointwise
ops. Composable with FSDP and other distributed strategies.

**Limitation**: Uses local cost heuristics (per-op compute/memory ratio). Cannot reason
about offloading, compression, or global scheduling effects. No awareness of pipeline
position or hardware topology.

---

## 3. The Frontier: Compression, Offloading, PyTorch APIs

### 3.1 Low-Rank Activation Compression

**LoRAct** (Shi et al., arXiv 2509.23472, 2025): Sampling-based orthogonal decomposition
compresses activations online. At rank r = d/8, achieves ~80% activation memory reduction.
Error bound: E‖A - Ã‖ ≤ (1 + C√(μ_k·k/ℓ))·σ_{k+1}(A) + k·exp(-ℓ/(μ_k·k)/C)·‖A‖.

**BOOST** (Wang et al., arXiv 2512.12131, Dec 2025): Bottleneck-aware tensor parallelism +
low-rank checkpointing. Aligns checkpoint boundaries with TP chunks to keep recomputation
local. 1.46-1.91× speedup over full-rank baselines.

**CompAct** (NAACL 2025): Random projections for 17-50% peak memory savings.

**PRAC** (arXiv 2602.23111, 2026): Provably minimum-variance unbiased gradient estimator
via principal-random subspace decomposition.

**COAT** (ICLR 2025): Compresses both optimizer states and activations to FP8.

### 3.2 Offloading to CPU and NVMe

**SSDTrain** (Wu et al., arXiv 2408.10013, 2024): Offloads activations to NVMe SSDs via
GPUDirect Storage. Tensor deduplication + adaptive Recompute-Offload-Keep Pareto curve.
47% peak activation memory reduction with negligible throughput loss. Compatible with
PyTorch, Megatron-LM, DeepSpeed.

**TERAIO** (NeurIPS 2025): Lifetime-aware tensor analysis — only 1.7% of GPU memory holds
active tensors on average. 1.47× speedup over ZeRO-Offload via GPUDirect Storage.

**GreedySnake** (arXiv 2512.17570, 2025): Vertical scheduling (process all micro-batches
per layer before advancing) for 1.96-2.53× throughput over ZeRO-Infinity.

**ZeRO-Offload / ZeRO-Infinity** (Microsoft): Offloads optimizer states and gradients to
CPU. ZeRO-Infinity extends to NVMe. Foundational but not optimized for selective
per-tensor decisions.

### 3.3 Other Notable Recent Work

**RevFFN** (arXiv 2512.20920, 2025): Reversible transformer blocks — reconstruct
activations from outputs, halving peak memory for MoE models.

**BurstEngine** (SC 2025): Sequence-level selective checkpointing for 1M+ token training.

**SimpleFSDP** (arXiv 2411.00284, 2024): FSDP via pure PyTorch primitives including SAC,
enabling full-graph torch.compile tracing.

### 3.4 Profiling Tools

**PyTorch Memory Profiler** (primary tool for this project):
```python
# Record full allocation history with stack traces
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... run training step ...
torch.cuda.memory._dump_snapshot("snapshot.pickle")
# Visualize at pytorch.org/memory_viz
```
Categorizes memory into: Parameters, Optimizer State, Gradients, **Activations**,
Temporary, Autograd Detail. Shows per-operator allocation/deallocation timeline.

**torch.profiler with memory tracking**:
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
) as prof:
    # ... training step ...
prof.export_memory_timeline("memory_timeline.html")
```

**Megatron-LM analytical formulas**: Use the Korthikanti formulas (34sbh/t for selective,
sbh(34+5as/h)/t for full storage) for memory planning at any scale without running the model.

**NVIDIA Nsight Systems**: System-level GPU/CPU activity, CUDA memory ops, NCCL
communication. Operates at CUDA level, not PyTorch semantic level.

**NVIDIA Nsight Compute**: Kernel-level memory access analysis for optimizing individual
recomputation kernels.

---

## 4. Algorithmic Proposals (Novel Contributions)

### 4.1 Proposal: Three-Resource Joint Optimization via Transformer-Aware DP

**Observation**: Remat, offload, and compression consume independent physical resources:
- **Recomputation** → GPU FLOPS
- **Offloading** → PCIe bandwidth
- **Compression** → error tolerance budget (+ small compute for decomposition)

No existing system optimizes over all three jointly. The Pareto frontier of
memory-vs-overhead should be substantially better when all three are co-optimized.

**Concrete example** (per transformer layer with FlashAttention active):
- Attention QK^T intermediates: already handled by FA → N/A
- MLP first linear input (8sbh): expensive matmul recompute, high rank → **OFFLOAD**
- MLP GeLU input (8sbh): expensive recompute, empirically low rank → **COMPRESS**
- LayerNorm inputs (4sbh): small, cheap → **KEEP**
- Dropout masks (2sbh): cheap to recompute → **RECOMPUTE**

**Why transformers enable polynomial-time solutions**:

On general DAGs, the joint remat+offload problem is strongly NP-hard (Beaumont et al.).
But transformers have a **series-parallel DAG** structure:

```
x → LN → Attn → (+) → LN → MLP → (+) → output
│               ↑    │              ↑
└── residual ───┘    └── residual ──┘
```

The residual connections are the only non-linearity. This creates exactly three liveness
categories for activation tensors:

1. **Intra-block intermediates** (QKV, softmax, GeLU inputs): born and die within one
   block's forward-backward cycle. Short liveness gap (~2× block compute time).

2. **Block boundary activations** (residual stream x): born at block entry, survive until
   this block's backward pass. Longest liveness gap — spans the remaining forward plus
   backward back to this point.

3. **Cross-block artifacts** (LN statistics, dropout masks): small tensors, long liveness,
   tiny memory footprint.

This decomposition is exact (not an approximation) and enables factoring the optimization:
- **Within-block decisions**: for ~10-15 intermediates per block, choose KEEP/RECOMPUTE/OFFLOAD/COMPRESS
- **Cross-block decisions**: for each of L boundaries, choose what to stash

#### 4.1.1 The DP Formulation

For each block i ∈ {1,...,L}, define a **block strategy** σ_i that specifies each
intermediate tensor's fate. Pre-compute a library of valid strategies per block (bounded
number due to fixed internal structure).

Each strategy σ_i has:
- mem(σ_i): peak HBM during block i's forward+backward
- compute(σ_i): additional FLOPs from recomputation
- pcie(σ_i, t): PCIe bandwidth consumed at time t (async transfers)
- error(σ_i): approximation error from compression

**DP recurrence** over state (block_index, cumulative_stashed_memory, pcie_in_flight, accumulated_error):

```
dp[i][m][p][e] = min compute cost to process blocks i..L
                  given m bytes stashed from earlier blocks,
                  p bytes of pending PCIe transfers,
                  e accumulated compression error

dp[i][m][p][e] = min over strategies σ_i of:
    compute(σ_i) + dp[i+1][m + stash(σ_i)][p'(σ_i)][e + error(σ_i)]
```

**PCIe state discretization**: Because blocks execute sequentially and each has known
compute duration, you can determine which prior offload transfers have completed by
block i's start. PCIe state reduces to "bytes still in transit" — discretize to ~50 bins.

**Error budget discretization**: 10-20 bins.
**Memory state discretization**: ~100 bins (bounded by HBM capacity).

**Total state space**: L × 100 × 50 × 20 = **100,000L** → trivially solvable in <1 second
for L ≤ 200. This is O(L) and produces the **globally optimal** joint remat+offload+compress
schedule for the transformer's specific DAG structure.

**No existing system does this.**

### 4.2 Proposal: Liveness-Aware Tensor Triage

Formalize which strategy is optimal for each tensor based on intrinsic properties.

Define per-tensor:
- λ(a_i): **liveness gap** — wall-clock time from creation in forward to last use in backward
- ρ(a_i): **recompute ratio** — FLOPs to recompute / original forward FLOPs
- μ(a_i): **size** in bytes
- κ(a_i): **effective rank** — fraction of energy in top-k singular values
- β_PCIe: available PCIe bandwidth
- β_GPU: available GPU compute bandwidth (spare FLOPs during forward)

**Decision rules (priority ordering)**:

1. **RECOMPUTE** when:
   - ρ(a_i) is small (cheap to recompute)
   - λ(a_i) · β_GPU > FLOPs_to_recompute (GPU has spare cycles)
   - Best candidate: attention softmax/dropout (ρ ≈ 0.02)

2. **OFFLOAD** when:
   - λ(a_i) · β_PCIe > 2·μ(a_i) (liveness gap covers round-trip transfer)
   - ρ(a_i) is large (too expensive to recompute)
   - Best candidate: MLP intermediates in deep networks (λ ≥ 50ms on A100,
     enough for ~800MB at 16GB/s PCIe 4.0)

3. **COMPRESS** when:
   - κ(a_i) is low (strong low-rank structure)
   - ρ(a_i) is large (expensive to recompute)
   - λ(a_i) · β_PCIe < 2·μ(a_i) (insufficient bandwidth to offload)
   - Best candidate: MLP activations with empirically low effective rank

4. **KEEP** when:
   - μ(a_i) is small, or all other options have high cost
   - Best candidate: LN statistics, dropout masks, small residual tensors

The DP from 4.1 finds the globally optimal assignment; this triage provides interpretable
insight into WHY that assignment is optimal and serves as a heuristic for cases where
the DP is too expensive.

### 4.3 Proposal: Pipeline-Position-Aware AC

**Novel observation not in any existing paper.**

Under 1F1B pipeline parallelism, a stage at position s from the end stashes (s-1)
in-flight microbatch activations. The memory cost of stashing is:

```
stash_cost(stage, position_s) = (s - 1) × activations_per_microbatch(AC_mode)
```

For PP=8, the first stage (s=7) pays **7×** the stashing memory of the last stage (s=0).

**Implication**: Stages near the pipeline's beginning should checkpoint more aggressively
because each saved byte has a (s-1)× memory multiplier:

- **Stage 0 (first, s=PP-1)**: Full recomputation + aggressive offloading. Highest memory
  pressure from stashing. Every byte of activation saved reduces peak memory by PP-1 bytes.
- **Stage PP-1 (last, s=0)**: No checkpointing. Zero stashed data, so keeping everything
  in HBM maximizes throughput with no memory penalty.
- **Intermediate stages**: Selective recomputation proportional to (s-1).

**No auto-parallelism system accounts for this.** NEST treats AC as binary per-stage.
Megatron-LM applies uniform AC. This is a free improvement requiring zero additional
hardware — just adapt the DP from 4.1 with a stage-dependent memory constraint:

```
Memory_budget(stage_s) = HBM - params - optimizer - (s-1) × stash_per_microbatch
```

where stash_per_microbatch itself depends on the AC strategy chosen for that stage.

### 4.4 Proposal: Offload Scheduling as Interval Scheduling

Current offload systems use greedy heuristics. The optimal problem is a classical
**weighted job scheduling on a single machine with release times and deadlines**.

**Formulation**: For each tensor a_j decided for offloading:
- Release time r(a_j): when created (end of producing op in forward)
- Deadline d(a_j): when needed back (start of consuming op in backward)
- Transfer time: t(a_j) = 2·μ(a_j)/β_PCIe (round-trip)

**Goal**: Maximize total memory freed by offloading, subject to PCIe bus capacity.

**Key property**: Tensors created earlier in forward have earlier deadlines in backward
(backward reverses forward order), giving "agreeable deadlines" — a known property that
makes the scheduling problem solvable in polynomial time.

**Optimal algorithm**: Process tensors in reverse liveness-gap order (longest gap first),
greedily assign to available PCIe time slots. O(n log n).

**No existing paper formalizes offloading this way.** SSDTrain uses ad-hoc Pareto curves;
TERAIO uses tensor lifetime analysis without scheduling theory guarantees.

### 4.5 Proposal: FA-Aware Activation Memory Formulas

Re-derive Korthikanti's formulas under FlashAttention:

**Without FA (Korthikanti)**:
```
Per-layer = sbh(34 + 5as/h) / t     [no AC]
Per-layer = 34sbh / t                [selective AC: recompute attention]
```

**With FA**:
```
Per-layer attention = 9sbh / t       [Q, K, V, output proj input, dropout mask]
Per-layer MLP = 19sbh / t            [LN input, GeLU input, linear2 input, dropout]
Per-layer total = 32sbh / t          [slightly less than 34 due to FA's reduced storage]
```

**The bottleneck shifts**: With FA, the MLP block (19sbh/t) now dominates the attention
block (9sbh/t). The optimal selective strategy shifts from "recompute attention, keep MLP"
to "keep attention (FA handles it), selectively recompute/offload/compress MLP."

Specifically, the 8sbh GeLU input and 8sbh second linear input are the new targets for
optimization — exactly the tensors where compression (low rank) and offloading (long
liveness gap in deep networks) are most effective.

### 4.6 Putting It Together: The Paper

**Title**: "Three-Resource Activation Memory Optimization for Transformer Training"

**Contributions**:
1. (Theory) Formalize joint remat+offload+compress. Show NP-hardness on general DAGs,
   O(L) DP on transformer series-parallel structure.
2. (Algorithm) Implement DP with three discretized state dimensions. Pre-compose
   within-block strategies. Integrate interval scheduling for offload timing.
3. (Pipeline) Extend DP for pipeline-position-aware AC via (s-1) stashing multiplier.
4. (FA-aware) Re-derive memory formulas under FlashAttention, identify shifted bottleneck.
5. (Evaluation) Beat Megatron-LM heuristic, PyTorch SAC, Checkmate (where it scales),
   and Beaumont's DP (extended to transformers). Show gains on 7B-175B models.

---

## 5. SDPO + TTT-Discover Setup

This section describes the optional follow-up: using reinforcement learning to learn
a checkpointing policy that approximates or extends the DP from Section 4.

### 5.1 Why SDPO suits this problem

**SDPO** (Self-Distillation Policy Optimization, Hübotter et al., arXiv 2601.20802, 2026)
provides dense per-token credit assignment via self-distillation. The key loss:

```
L_SDPO(θ) = Σ_t KL(π_θ(·|x, y_{<t}) ‖ stopgrad(π_θ(·|x, f, y_{<t})))
```

where f is rich feedback. Per-token advantages:
```
A^SDPO_t(ŷ_t) = log π_θ(ŷ_t | x, f, y_{<t}) - log π_θ(ŷ_t | x, y_{<t})
```

In practice uses JSD (α=0.5), EMA teacher (τ=0.05), top-K logit distillation (K=100).

**Why it fits**: The memory profiler produces rich structured feedback (per-GPU memory
waterfall, per-layer recompute times, PCIe utilization, pipeline bubble fraction) that
SDPO's self-teacher can use for dense credit assignment — identifying exactly which
per-layer decisions caused memory overflow or wasted compute.

### 5.2 TTT-Discover at compile time

**TTT-Discover** (Yuksekgonul et al., arXiv 2601.16175, 2026) adapts model weights via RL
during inference using an entropic utility objective:

```
J_β(θ) = E_{s~PUCT(H)} [log E_{a~π_θ(·|s)} [e^{β(s)·R(s,a)}]]
```

As β→∞, converges to max_a R(s,a). Combined with PUCT-based state reuse from AlphaZero.

**Application**: When a new model arrives for training, run TTT-Discover with the
SDPO-pretrained policy for ~50 LoRA-based RL steps against the memory simulator. The
entropic objective targets the single best AC schedule (not the average-case schedule).

### 5.3 Base Model Selection

**Recommended**: Qwen2.5-Coder-7B-Instruct or Qwen3-Coder-8B.

**Reasoning**:
- Structured output (config generation) is fundamentally a code task
- 7-8B fits comfortably in a single GPU for inference — critical for generating 64-512
  rollouts per SDPO training step
- Strong at arithmetic reasoning (divisibility, memory budgets)
- Apache 2.0 licensed
- If too weak: scale to Qwen2.5-32B with LoRA before trying larger models

**Do not use**: gpt-oss-120b (MoE, designed for inference not finetuning), models >32B
(too slow for rollout generation).

### 5.4 Hardware Requirements

**For SDPO finetuning of 7B policy model**:
- 4× A100-80GB (or 4× H100-80GB)
- With LoRA rank 64: can drop to 2× A100-80GB
- Cloud cost: ~$8-12/hr on Lambda Labs / CoreWeave
- Budget: ~$2,000-5,000 for full Phase 1 training (~200-500 GPU-hours)

**For the simulator**: CPU only. No GPUs needed for the analytical cost model.

**For validation on real hardware**: 1-8 GPUs for profiling actual training steps.
Calibration only — not in the RL loop.

### 5.5 The Simulator (RL Environment)

The simulator takes (model_spec, hardware_spec, AC_schedule) and returns
(throughput, rich_feedback).

**Implementation**: Python analytical model computing per-layer:
- Activation memory per the Korthikanti formulas (with FA modifications)
- Recompute cost from profiled per-operator latencies (one-time measurement)
- Offload cost: tensor_size / PCIe_bandwidth
- Overlap feasibility: check if offload transfer fits within next layer's compute time
- Peak memory across the full forward-backward pass

**Speed target**: 1000+ evaluations/second (essential for RL training feasibility).

**Calibration**: Validate against PyTorch memory profiler measurements on real hardware.
Target <5% error on peak memory prediction, <10% error on throughput prediction.

### 5.6 Input/Output Format

**Input prompt** (what the policy model receives):
```
<model>
  architecture: transformer
  hidden_dim: 4096, num_layers: 32, num_heads: 32, kv_heads: 8
  intermediate_dim: 11008
  sequence_length: 4096, micro_batch_size: 2
  dtype: bf16, flash_attention: true
</model>
<hardware>
  gpu: A100-80GB, num_gpus: 1
  hbm: 80GB, pcie_bandwidth: 32GB/s
  gpu_bf16_tflops: 312
</hardware>
<memory_budget>72GB (90% of HBM)</memory_budget>
<task>For each layer, decide: KEEP, CHECKPOINT, OFFLOAD_CPU, or COMPRESS</task>
```

**Output** (what the model generates):
```
layer_0:  {attn: KEEP, mlp_gelu: COMPRESS_r8, mlp_linear2: OFFLOAD, ln: KEEP, dropout: RECOMPUTE}
layer_1:  {attn: KEEP, mlp_gelu: COMPRESS_r8, mlp_linear2: OFFLOAD, ln: KEEP, dropout: RECOMPUTE}
...
layer_31: {attn: KEEP, mlp_gelu: KEEP, mlp_linear2: KEEP, ln: KEEP, dropout: KEEP}
```

**Rich feedback** (for SDPO self-teacher):
```
<feedback>
  throughput: 1247 tokens/sec
  peak_memory: 74.2GB / 72GB budget (EXCEEDED by 2.2GB!)
  peak_memory_timestep: backward pass of layer 14
  per_layer_memory_at_peak: [layer_0: 2.1GB, layer_1: 2.1GB, ..., layer_14: 3.8GB, ...]
  recompute_overhead: 2.3% (dropout masks across 32 layers)
  offload_stalls: layer_3 mlp_linear2 stalled GPU for 1.2ms (transfer 67MB, only 41ms compute available)
  compression_error: accumulated 0.0023 (within budget of 0.01)
  diagnosis: Peak memory exceeded budget. Layer 14's KEEP on mlp_linear2 (67MB) pushed peak over.
             Consider OFFLOAD or COMPRESS for layers 12-16 to reduce peak by ~200MB.
</feedback>
```

### 5.7 Training Data Generation

**Problem instances** (generated procedurally, not collected):
- Model architectures: hidden_dim ∈ {2048, 4096, 5120, 8192, 12288},
  num_layers ∈ {24, 32, 48, 64, 80, 96}, with/without GQA, with/without MoE
- Hardware configs: A100-40GB, A100-80GB, H100-80GB, with varying PCIe gen
- Memory budgets: 60-95% of HBM
- Yields ~5,000-50,000 unique instances

**Supervised warm-start data**:
- Run the DP from Section 4 on small instances → optimal labels
- Run Megatron-LM heuristic, PyTorch SAC, √n checkpointing → suboptimal baselines
- Run brute force on very small instances (≤12 layers)
- Use for 1-2 epochs of SFT before switching to SDPO

### 5.8 Evaluation Protocol

**Baselines to compare against**:
1. Checkpoint-nothing (max memory, no overhead)
2. Checkpoint-everything (min memory, ~33% overhead)
3. Megatron-LM selective (checkpoint attention only — current industry standard)
4. PyTorch SAC Memory Budget API (min-cut/knapsack)
5. Checkmate ILP (optimal for small models, intractable for large)
6. √n heuristic (evenly spaced)
7. The DP from Section 4 (the algorithmic contribution, no RL)

**Metrics**: Peak memory, training throughput (tokens/sec), recompute overhead (%),
compression error, Pareto frontier of memory vs. throughput.

**Success criteria**:
- Match DP optimality on small models where DP can run
- Beat Megatron-LM heuristic by measurable margin on unseen architectures
- Generalize across model families (train on GPT, test on Llama) via TTT-Discover

---

## 6. Practical Implementation Plan

### 6.1 Timeline

**Phase 1: Algorithmic contribution (Weeks 1-6)**
- Weeks 1-2: Build analytical memory simulator. Implement Korthikanti formulas with
  FA-aware modifications. Validate against PyTorch memory profiler on real hardware.
- Weeks 3-4: Implement the three-resource DP (Section 4.1). Define block strategy library.
  Implement interval scheduling for offload timing.
- Weeks 5-6: Extend DP for pipeline-position awareness. Run evaluation on GPT/Llama/Mixtral
  architectures. Compare against all baselines.

**Phase 2: SDPO+TTT extension (Weeks 7-12)**
- Weeks 7-8: Implement input/output format. Generate problem instances. SFT warm-start
  using DP solutions + heuristic baselines.
- Weeks 9-10: Implement SDPO training loop. Debug feedback format, reward signal, training
  stability. Phase 1 training campaign (~200-500 GPU-hours).
- Weeks 11-12: TTT-Discover implementation. Evaluation on held-out configs. Ablations.

### 6.2 Key Implementation Details

**Simulator validation target**: <5% error on peak memory vs. PyTorch profiler,
<10% error on throughput vs. real training.

**SDPO hyperparameters** (from the paper):
- EMA teacher momentum: τ = 0.05
- Top-K logit distillation: K = 100
- JSD divergence: α = 0.5
- Learning rate: ~1e-5 (full finetune) or ~2e-4 (LoRA rank 64)
- Rollouts per step: K = 8
- Batch size: 16 problem instances

**TTT-Discover hyperparameters** (from the paper):
- LoRA rank: 32
- Training steps: 50
- Rollouts per step: 512
- Adaptive β per state (start ~1.0, increase to ~10.0)
- PUCT exploration constant c: tune on validation set

### 6.3 Repository Structure (Suggested)

```
selective-ac-optimizer/
├── README.md
├── KNOWLEDGE_BASE.md              ← this file
├── simulator/
│   ├── memory_model.py            ← Korthikanti formulas, FA-aware
│   ├── compute_model.py           ← per-operator FLOPs and latencies
│   ├── offload_model.py           ← PCIe transfer scheduling
│   ├── compression_model.py       ← low-rank error estimates
│   └── environment.py             ← RL environment wrapping the above
├── dp_solver/
│   ├── block_strategies.py        ← enumerate valid within-block strategies
│   ├── dp_solver.py               ← three-resource DP (Section 4.1)
│   ├── interval_scheduler.py      ← optimal offload scheduling (Section 4.4)
│   └── pipeline_aware.py          ← position-dependent memory constraints (Section 4.3)
├── baselines/
│   ├── sqrt_n.py                  ← Chen et al. uniform segmentation
│   ├── megatron_selective.py      ← Korthikanti hard-coded attention recompute
│   ├── pytorch_sac.py             ← PyTorch Memory Budget API wrapper
│   └── checkmate_wrapper.py       ← Checkmate ILP (for small-model validation)
├── learned_policy/
│   ├── data_generation.py         ← procedural problem instance generation
│   ├── sft_warmstart.py           ← supervised finetuning on DP solutions
│   ├── sdpo_trainer.py            ← SDPO training loop
│   ├── ttt_discover.py            ← test-time adaptation
│   └── prompts.py                 ← input/output format templates
├── evaluation/
│   ├── profiling.py               ← PyTorch memory profiler integration
│   ├── benchmark.py               ← run all baselines + learned policy
│   └── pareto.py                  ← memory-throughput Pareto analysis
└── configs/
    ├── models/                    ← model architecture specs (GPT, Llama, Mixtral, ...)
    └── hardware/                  ← hardware profiles (A100-40, A100-80, H100-80, ...)
```

---

## References

### Foundational Papers
- Chen et al., "Training Deep Nets with Sublinear Memory Cost," arXiv 1604.06174, 2016
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," arXiv 2205.05198, MLSys 2023
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention," NeurIPS 2022

### Optimal Scheduling Algorithms
- Jain et al., "Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization," MLSys 2020
- Shah et al., "MONeT: Memory Optimization for Deep Networks," ICLR 2021
- Beaumont et al., "Efficient Combination of Rematerialization and Offloading for Training DNNs," NeurIPS 2021
- Kirisame et al., "Dynamic Tensor Rematerialization," ICLR 2021
- Bartan et al., "MOCCASIN: Efficient Tensor Rematerialization for Neural Networks," ICML 2023

### Compression Methods
- Shi et al., "LoRAct: Low-Rank Activation Compression," arXiv 2509.23472, 2025
- Wang et al., "BOOST: Bottleneck-Optimized Scalable Training," arXiv 2512.12131, 2025
- Shamshoum et al., "CompAct: Compressed Activations for Memory-Efficient LLM Training," NAACL 2025
- "COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training," ICLR 2025

### Offloading Systems
- Wu et al., "SSDTrain: Activation Offloading to SSDs," arXiv 2408.10013, 2024
- "TERAIO: Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage," NeurIPS 2025
- "GreedySnake: Accelerating SSD-Offloaded LLM Training," arXiv 2512.17570, 2025

### RL / LLM for Systems
- Hübotter et al., "SDPO: Reinforcement Learning via Self-Distillation," arXiv 2601.20802, 2026
- Yuksekgonul et al., "Learning to Discover at Test Time," arXiv 2601.16175, 2026
- Cummins et al., "Meta Large Language Model Compiler," 2024

### Auto-Parallelism (Context)
- Wang et al., "NEST: Network- and Memory-Aware Device Placement," arXiv 2603.06798, 2026
- Zheng et al., "Alpa: Automating Inter- and Intra-Operator Parallelism," OSDI 2022
- Narayanan et al., "Megatron-LM: Efficient Large-Scale Training," SC 2021

### PyTorch APIs
- PyTorch SAC Memory Budget API: torch.compile with activation_memory_budget
- PyTorch Memory Profiler: torch.cuda.memory._record_memory_history / _dump_snapshot
- IBM/Meta FSDP Selective AC: pytorch.org/blog/maximizing-training/
