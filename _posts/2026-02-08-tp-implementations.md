---
title: "Tensor Parallel 구현 비교"
date: 2026-02-08
last_modified_at: 2026-02-08
layout: post
permalink: /blog/tp-implementations/
description: "Megatron-LM, nanotron, DeepSpeed, torchtitan, vLLM, SGLang의 Tensor Parallelism 구현을 코드 레벨에서 비교 분석합니다."
tags: tensor-parallelism distributed-training implementation-comparison
thumbnail: assets/img/blog/tp-impl.png
series: distributed-training
series_order: 2
series_title: "Distributed Training Series"
related_posts: true
disqus_comments: false
giscus_comments: true
toc:
  sidebar: left
---

# TL;DR

> 이 글은 [Tensor Parallel 이론편](/blog/tp/)의 후속으로, 6개 주요 프레임워크의 실제 TP 구현을 비교합니다.
> 학습 프레임워크(Megatron-LM, nanotron, DeepSpeed, torchtitan)와 추론 프레임워크(vLLM, SGLang)는 동일한 이론적 기반 위에서 서로 다른 최적화 철학을 보여줍니다.
> Megatron-LM의 f/g 켤레 연산자 패러다임은 거의 모든 프레임워크에 영향을 주었으나, 각자의 요구사항에 맞게 변형되었습니다.
> 핵심 차이점: 학습 프레임워크는 그래디언트 통신 최적화에, 추론 프레임워크는 weight loading과 레이턴시 최소화에 집중합니다.

---

# Related Work

**이론적 기반:**
- [Tensor Parallel 이론편](/blog/tp/) - Megatron-LM 논문의 핵심 아이디어
- [Pipeline Parallel](/blog/pp/) - 레이어 간 병렬화
- [Sequence Parallel](/blog/sp/) - 시퀀스 차원 병렬화

**프레임워크 문서:**
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - NVIDIA의 대규모 Transformer 학습
- [nanotron](https://github.com/huggingface/nanotron) - HuggingFace의 분산 학습 라이브러리
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Microsoft의 분산 학습 최적화
- [torchtitan](https://github.com/pytorch/torchtitan) - Meta의 PyTorch-native 학습 플랫폼
- [vLLM](https://github.com/vllm-project/vllm) - 고성능 LLM 서빙 엔진
- [SGLang](https://github.com/sgl-project/sglang) - 구조화된 출력 지원 LLM 서빙

---

# 1. 서론: 왜 구현을 비교하는가?

[이전 글](/blog/tp/)에서 Megatron-LM이 제안한 Tensor Parallelism의 이론적 기반을 살펴보았습니다. 핵심 아이디어는 단순합니다:

1. **ColumnParallelLinear**: 출력 차원을 분할하고, backward에서 all-reduce
2. **RowParallelLinear**: 입력 차원을 분할하고, forward에서 all-reduce
3. **f/g 켤레 연산자**: forward와 backward에서 상호 보완적인 통신 패턴

그러나 실제 구현에서는 다양한 트레이드오프가 존재합니다:

| 고려 사항 | 학습 최적화 | 추론 최적화 |
|----------|-----------|-----------|
| **주요 병목** | 그래디언트 통신 | weight loading, 레이턴시 |
| **메모리 관심** | 활성화 + optimizer 상태 | KV cache |
| **배치 크기** | 크게 (수천) | 작게 (수십~수백) |
| **통신 패턴** | 비동기 오버랩 중요 | 동기 단순성 선호 |

이 글에서는 6개 프레임워크의 실제 코드를 분석하여, 이론이 어떻게 다양한 형태로 구현되는지 살펴봅니다.

---

# 2. 학습 프레임워크

## 2.1 Megatron-LM: 원조의 정석

**핵심 파일:** `megatron/core/tensor_parallel/`

Megatron-LM은 TP의 원조답게 가장 정교한 구현을 제공합니다. 핵심은 **f/g 켤레 연산자**입니다.

### 핵심 연산자 4종 세트

Megatron-LM은 4가지 핵심 autograd 연산자를 정의합니다:

| 연산자 | Forward | Backward | 용도 |
|--------|---------|----------|------|
| `_CopyToModelParallelRegion` | identity | all-reduce | ColumnParallel 입력 |
| `_ReduceFromModelParallelRegion` | all-reduce | identity | RowParallel 출력 |
| `_ScatterToModelParallelRegion` | split(last) | gather(last) | hidden dim 분할 |
| `_GatherFromModelParallelRegion` | gather(last) | split(last) | hidden dim 수집 |

**파일:** `megatron/core/tensor_parallel/mappings.py` (lines 197-273)

### f 연산자: Copy to TP Region

```python
# megatron/core/tensor_parallel/mappings.py:197-214
class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        """Forward: identity (통신 없음)"""
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: all-reduce gradients"""
        return _reduce(grad_output, ctx.group), None
```

**용도:** ColumnParallelLinear 시작 부분에서 사용. 입력 $X$는 모든 랭크에 복제되어 있으므로 forward에서는 그대로 전달하고, backward에서 각 랭크의 부분 그래디언트를 합산합니다.

### g 연산자: Reduce from TP Region

```python
# megatron/core/tensor_parallel/mappings.py:217-233
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        """Forward: all-reduce across TP ranks"""
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: identity (통신 없음)"""
        return grad_output, None
```

**용도:** RowParallelLinear 끝 부분에서 사용. 각 랭크가 계산한 $Y_i = X_i A_i$를 합산하여 최종 출력 $Y = \sum_i Y_i$를 생성합니다.

### Scatter/Gather 연산자

Scatter와 Gather는 hidden dimension을 분할/수집할 때 사용됩니다:

```python
# megatron/core/tensor_parallel/mappings.py:236-273
class _ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        """Forward: split along last dim"""
        ctx.group = group
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: gather along last dim"""
        return _gather_along_last_dim(grad_output, ctx.group), None

class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        """Forward: gather along last dim"""
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: split along last dim"""
        return _split_along_last_dim(grad_output, ctx.group), None
```

### 비동기 통신-계산 오버랩

Megatron-LM의 핵심 혁신 중 하나는 **통신과 계산의 오버랩**입니다:

```python
# megatron/core/tensor_parallel/layers.py:494-627
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        tp_group = ctx.tp_group

        # 1. 입력 그래디언트 계산
        grad_input = grad_output.matmul(weight)

        # 2. 비동기 all-reduce 시작 (통신)
        if ctx.allreduce_dgrad:
            handle = torch.distributed.all_reduce(
                grad_input, group=tp_group, async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        # 3. 가중치 그래디언트 계산 (계산) - 통신과 동시에!
        if ctx.gradient_accumulation_fusion:
            # CUDA 커널로 main_grad에 직접 누적 → 중간 텐서 할당 제거
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_output, weight.main_grad
                )
            # ...
        else:
            grad_weight = grad_output.t().matmul(total_input)

        # 4. all-reduce 완료 대기
        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, ...
```

**핵심 통찰:**
- `async_op=True`로 all-reduce를 비동기로 시작
- `CUDA_DEVICE_MAX_CONNECTIONS=1` 환경 변수로 CUDA 스트림 스케줄링 순서 보장
- `fused_weight_gradient_mlp_cuda` 커널로 그래디언트를 `weight.main_grad`에 직접 누적 → 중간 텐서 할당 제거

### VocabUtility 패딩 전략

어휘 크기를 TP 그룹 간 균등하게 분할합니다:

```python
# megatron/core/tensor_parallel/utils.py:97-121
class VocabUtility:
    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        """Vocab range from global vocab size."""
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l
```

**왜 필요한가:** TP 그룹 간 균등한 어휘 분할 + all-gather/reduce-scatter 효율성

### Vocab Parallel Cross-Entropy

임베딩 레이어의 출력 통신을 최소화하기 위해, cross-entropy 손실을 병렬로 계산합니다:

```python
# megatron/core/tensor_parallel/cross_entropy.py
class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        # 1. logits_max: 수치 안정성을 위한 all-reduce (MAX)
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max, op=ReduceOp.MAX)

        # 2. predicted_logits: 타겟 토큰의 logit만 all-reduce (SUM)
        # (각 랭크는 자신의 vocab 파티션에 해당하는 타겟만 처리)

        # 3. sum_exp_logits: 분할 함수 계산을 위한 all-reduce (SUM)

        # 최종: loss = log(sum_exp) - predicted_logits
```

**통신량:** $O(b \times s)$ (vocab 차원 제거) vs $O(b \times s \times v)$ (전체 logits 수집)

---

## 2.2 nanotron: 모듈러 설계

**핵심 파일:** `src/nanotron/parallel/tensor_parallel/`

nanotron은 HuggingFace에서 개발한 분산 학습 라이브러리로, Megatron-LM의 설계를 더 모듈러하게 재구성했습니다.

### 두 가지 TP 모드: ALL_REDUCE vs REDUCE_SCATTER

nanotron의 가장 큰 특징은 **명시적인 2가지 TP 모드**를 지원한다는 점입니다:

```python
# src/nanotron/parallel/tensor_parallel/enum.py
class TensorParallelLinearMode(Enum):
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
```

**ALL_REDUCE 모드 (lines 248-251):**
```python
if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
    gathered_tensor = tensor  # 통신 없음, 입력이 이미 복제됨
    return F.linear(gathered_tensor, weight, bias)
```

**REDUCE_SCATTER 모드 (lines 252-380):**
```python
elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
    # Forward: AllGather
    handle = dist.all_gather_into_tensor(gathered_tensor, tensor, group=group, async_op=True)
    # Backward: ReduceScatter
```

**언제 사용하는가:**
- **ALL_REDUCE**: 입력이 복제된 경우 (표준 TP)
- **REDUCE_SCATTER**: 입력이 배치 차원으로 분할된 경우 (SP와 결합 시)

### same_device_shard 최적화 패턴

AllGather 대기 중 자신의 데이터로 먼저 계산하여 통신을 오버랩합니다:

```python
# src/nanotron/parallel/tensor_parallel/functional.py:305-327
# AllGather 비동기 시작
handle = dist.all_gather_into_tensor(gathered_tensor, tensor, group=group, async_op=True)

# 출력 텐서를 before/same/after로 분할
before_shard, same_device_shard, after_shard = torch.split(
    gathered_output,
    split_size_or_sections=[
        sharded_batch_size * current_rank,
        sharded_batch_size,  # 자신의 데이터
        sharded_batch_size * (group_size - current_rank - 1),
    ],
    dim=0,
)

# AllGather 완료 전에 자신의 shard 먼저 계산
torch.mm(
    input=tensor.view(first_dims, hidden_size),
    mat2=weight.t(),
    out=same_device_shard.view(first_dims, output_size),
)

# AllGather 완료 대기
handle.wait()

# 나머지 shard 계산
if before_shard.numel() > 0:
    torch.mm(
        input=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
        mat2=weight.t(),
        out=before_shard.view(first_dims, output_size),
    )
# after_shard도 동일하게 처리
```

**성능 이점:** ~33% 계산이 통신과 오버랩 (TP=3일 때)

### tp_recompute_allgather 트레이드오프

nanotron의 독특한 메모리 최적화:

```python
# src/nanotron/parallel/tensor_parallel/nn.py
class TensorParallelColumnLinear(nn.Linear):
    def __init__(self, ..., tp_recompute_allgather=False):
        self.tp_recompute_allgather = tp_recompute_allgather
```

**Forward (메모리 절약):**
```python
if tp_recompute_allgather:
    gathered_tensor = MemoryBuffer().get("allgather", ...)  # 버퍼 재사용
    ctx.save_for_backward(tensor, weight)  # sharded tensor만 저장
```

**Backward (재계산):**
```python
if ctx.tp_recompute_allgather:
    unsharded_tensor = MemoryBuffer().get("allgather", ...)
    handle = dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group, async_op=True)
    # AllGather 다시 수행
```

**트레이드오프:**

| 설정 | 메모리 | 계산 |
|------|--------|------|
| `tp_recompute_allgather=False` | O(batch) | AllGather 1회 |
| `tp_recompute_allgather=True` | O(batch/TP) | AllGather 2회 |

### bias 처리 차이

- **ColumnParallel:** bias 분할됨 (`out_features/TP`)
- **RowParallel:** rank 0만 bias 보유 (`dist.get_rank(self.pg) == 0 and bias`)

### Contiguous Chunks

QKV 프로젝션처럼 여러 텐서를 하나로 퓨전할 때 유용:

```python
qkv_contiguous_chunks = (
    config.num_attention_heads * self.d_qk,      # Q 청크
    config.num_key_value_heads * self.d_qk,      # K 청크
    config.num_key_value_heads * self.d_qk,      # V 청크
)

self.qkv_proj = TensorParallelColumnLinear(
    hidden_size, q_out + 2*kv_out,
    contiguous_chunks=qkv_contiguous_chunks,
)
```

각 청크 내에서 독립적으로 TP 분할이 적용됩니다.

### ParallelContext: 5D 병렬화

nanotron은 5차원 병렬화를 지원합니다:

```python
# src/nanotron/parallel/context.py
# [expert_parallel, pipeline_parallel, data_parallel, context_parallel, tensor_parallel]

self.tp_pg         # TP group
self.dp_pg         # DP group
self.pp_pg         # PP group
self.cp_pg         # Context Parallel group (시퀀스 병렬화)
self.ep_pg         # Expert Parallel group (MoE용)
```

---

## 2.3 DeepSpeed: 자동화 추구

**핵심 파일:** `deepspeed/module_inject/`

DeepSpeed는 **자동 TP 적용**에 초점을 맞춥니다. 사용자가 모델 코드를 수정하지 않아도 TP를 적용할 수 있습니다.

### AutoTP GEM 리스트 탐지 로직

GEM(General Embedding/Matrix) 리스트는 all-reduce가 필요한 레이어를 자동 탐지합니다:

```python
# deepspeed/module_inject/auto_tp.py:307-338
def tp_parser(model):
    for module in module_list:
        for key, submodule in module._modules.items():
            if isinstance(submodule, nn.Linear):
                layer_list = layer_list + ["." + key]
            elif isinstance(submodule, nn.LayerNorm) or key in norm_layer_name_list:
                layer_list = layer_list + ["ln"]

        for i, layer in enumerate(layer_list):
            if layer == 'ln':
                if layer_list[i - 1] != 'ln':
                    gem_list.append(layer_list[i - 1])  # LN 직전 = all-reduce 필요
            elif 'out_proj' in layer:
                gem_list.append(layer)
            elif 'down_proj' in layer:
                gem_list.append(layer)
```

**GEM = General Embedding/Matrix:** LayerNorm 직전 레이어, `out_proj`, `down_proj` 등 all-reduce가 필요한 레이어 목록

### 8종 Fused QKV 포맷 핸들링

다양한 모델이 QKV를 다르게 퓨전합니다:

```python
# deepspeed/module_inject/fusedqkv_utils.py:34-46
fused_type_dict = {
    'CodeGenBlock': 'codegentype',   # [q(1),q(2),...,k(1),k(2),...,v(1),v(2),...]
    'BloomBlock': 'bloomtype',        # [q(1),k(1),v(1),q(2),k(2),v(2),...]
    'GLMBlock': 'glmtype',            # [Q,Q,...,K,K,...,V,V,...]
    "MPTBlock": 'glmtype',
    "BaichuanLayer": 'glmtype',
    "QWenBlock": 'qwentype',
    "FalconDecoderLayer": 'bloomtype',
    "GPTBigCodeBlock": 'bigcodetype',
    "Phi3DecoderLayer": "phi3type",   # Rotary 임베딩 위치 분리
}
```

| 모델 | 포맷 | 레이아웃 |
|------|------|---------|
| Bloom/Falcon | bloomtype | `[q1,k1,v1,q2,k2,v2,...]` (interleaved) |
| ChatGLM/MPT | glmtype | `[Q,Q,...,K,K,...,V,V,...]` (stacked) |
| CodeGen | codegentype | 멀티블록 레이아웃 |
| Phi3 | phi3type | Rotary 임베딩 위치 분리 |

DeepSpeed는 각 포맷에 맞는 전치/분할 로직을 자동 적용합니다.

### SubParamLinearLayer 불균등 파티셔닝

GQA(Grouped-Query Attention)처럼 Q와 KV 헤드 수가 다를 때:

```python
# deepspeed/module_inject/layers.py
class SubParamLinearLayer(TensorParallel_Layer):
    def __init__(self, module, mp_group, shape, partition_dim=0):
        # shape = ((q_size, k_size, v_size), -1) for GQA
        # 각 서브파라미터를 독립적으로 분할
```

**GQA 예시:**
```python
shape = ((4096, 1024, 1024), -1)  # Q: 4096, K: 1024, V: 1024
# 각 sub-param을 독립적으로 분할
sub_params = torch.split(tensor, subparam_sizes, dim=partition_dim)
partitioned = [torch.chunk(sp, tp_size, dim=0)[tp_idx] for sp in sub_params]
```

### tp_grain_size 불균등 분할

어휘 크기가 TP 크기로 나누어 떨어지지 않을 때:

```python
# deepspeed/module_inject/tp_shard.py:47-67
def get_shard_size(total_size, mp_size, name=None, rank=None):
    if total_size >= tp_grain_size:
        grain_size = total_size // tp_grain_size
        return (grain_size // mp_size + (1 if rank < (grain_size % mp_size) else 0)) * tp_grain_size
    else:
        return total_size // mp_size + (1 if rank < (total_size % mp_size) else 0)
```

**예시:** `total=4096, tp_size=8, grain_size=128` → 각 랭크 512 토큰 (균등 분할)

---

## 2.4 torchtitan: PyTorch Native

**핵심 파일:** `torchtitan/distributed/`

torchtitan은 PyTorch의 **DTensor**와 **DeviceMesh** API를 사용하여 TP를 구현합니다. 커스텀 autograd 함수 없이 선언적으로 병렬화를 정의합니다.

### 선언적 병렬화 계획

```python
# torchtitan/models/llama3/infra/parallelize.py:161-248
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput,
)

def apply_tp(model, tp_mesh, loss_parallel, enable_float8_tensorwise_tp, cp_enabled):
    # 1. 임베딩, 정규화, 출력 레이어 병렬화
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # 2. 각 Transformer 블록 병렬화
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None, None, None),      # 현재: sequence dim 분할
                desired_input_layouts=(Replicate(), None, None, None),  # 목표: 복제
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }
        parallelize_module(transformer_block, tp_mesh, layer_plan)
```

**핵심 차이:** 명시적 f/g 연산자 대신 DTensor가 레이아웃 변환을 자동 처리

### loss_parallel 컨텍스트 매니저

```python
# torchtitan/distributed/utils.py
if enable_loss_parallel:
    stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
```

**효과:**
- 출력 레이어가 `Shard(-1)` (vocab dim 분할) 유지
- all-gather 지연으로 메모리 절약
- cross-entropy 계산 시 자동 분산 처리

### Float8 TP 지원

```python
# torchtitan/models/llama3/infra/parallelize.py:192-210
if enable_float8_tensorwise_tp:
    from torchao.float8.float8_tensor_parallel import (
        Float8ColwiseParallel,
        Float8RowwiseParallel,
        PrepareFloat8ModuleInput,
    )
    rowwise_parallel, colwise_parallel, prepare_module_input = (
        Float8RowwiseParallel,
        Float8ColwiseParallel,
        PrepareFloat8ModuleInput,
    )
```

**제약:** Tensorwise float8만 TP 지원, rowwise는 표준 TP 사용

### 레이아웃 명세

```python
from torch.distributed.tensor import Replicate, Shard

Replicate()   # 전체 텐서를 모든 랭크에 복제
Shard(0)      # 0번 차원으로 분할 (batch)
Shard(1)      # 1번 차원으로 분할 (sequence/hidden)
Shard(-1)     # 마지막 차원으로 분할 (vocab dim)
```

### torch.compile 호환

DTensor 기반 구현은 `torch.compile`과 자연스럽게 통합됩니다:

```python
# Async TP 활성화
torch._inductor.config._micro_pipeline_tp = True
```

---

# 3. 추론 프레임워크

추론 프레임워크는 학습과 다른 최적화 방향을 가집니다:
- **그래디언트 없음:** backward pass 최적화 불필요
- **레이턴시 중심:** 단일 토큰 생성 시간이 중요
- **Weight Loading:** 모델 로딩 시 샤딩 적용

## 3.1 vLLM: 추론 최적화의 정석

**핵심 파일:** `vllm/distributed/`, `vllm/model_executor/layers/`

### GroupCoordinator 상세 구현

```python
# vllm/distributed/parallel_state.py:276-505
class GroupCoordinator:
    """PyTorch ProcessGroup wrapper for a group of processes."""

    def __init__(self, group_ranks, local_rank, torch_distributed_backend, ...):
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)  # custom op에서 조회할 수 있도록 등록

        # CPU와 device 통신 그룹 분리
        self.cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        self.device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if self.use_custom_op_call:
            return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
        else:
            return self._all_reduce_out_place(input_)
```

**핵심:** 단일 GPU 시 통신 스킵, custom op으로 torch.compile 지원

### Custom Op 등록

```python
# vllm/distributed/parallel_state.py:248-273
direct_register_custom_op(
    op_name="all_reduce",
    op_func=all_reduce,
    fake_impl=all_reduce_fake,  # torch.compile용 shape 추론
)

direct_register_custom_op(
    op_name="reduce_scatter",
    op_func=reduce_scatter,
    fake_impl=reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="all_gather",
    op_func=all_gather,
    fake_impl=all_gather_fake,
)
```

### Weight Loader 패턴

추론에서는 모델 로딩 시 weight를 분할합니다 (런타임 아님):

```python
# vllm/model_executor/layers/linear.py:551-586
class ColumnParallelLinear(LinearBase):
    def weight_loader(self, param, loaded_weight):
        output_dim = getattr(param, "output_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)

        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            # 로드 시점에 weight를 분할
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        param_data.copy_(loaded_weight)
```

**장점:** 런타임에 분할 연산 없음, 추론 레이턴시 최소화

### QKVParallelLinear GQA/MQA 처리

```python
# vllm/model_executor/layers/linear.py:954-962
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, ...):
        tp_size = get_tensor_model_parallel_world_size()

        if tp_size >= self.total_num_kv_heads:
            # KV 헤드 < TP 크기: KV 헤드 복제
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            # KV 헤드 >= TP 크기: KV 헤드 분할
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
```

GQA(Grouped-Query Attention)에서 KV 헤드 수가 Query 헤드보다 적을 때, 자동으로 복제/분할을 결정합니다.

### input_is_parallel / reduce_results 플래그

```python
# vllm/model_executor/layers/linear.py (RowParallelLinear)
def forward(self, input_):
    if self.input_is_parallel:
        input_parallel = input_  # scatter 스킵, 이미 분할되어 있음
    else:
        splitted = split_tensor_along_last_dim(input_, self.tp_size)
        input_parallel = splitted[self.tp_rank]

    output_parallel = self.quant_method.apply(self, input_parallel, bias)

    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)
```

ColumnParallel의 출력을 바로 RowParallel에 연결할 때, 중간 scatter를 스킵합니다.

---

## 3.2 SGLang: 통신 오케스트레이션

**핵심 파일:** `sglang/srt/layers/`

SGLang은 vLLM 기반이지만, 구조화된 출력(structured output) 지원과 함께 독자적인 최적화를 추가했습니다.

### ScatterMode 상세 설명

```python
# sglang/srt/layers/communicator.py:102-120
class ScatterMode(Enum):
    """
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively
    """
    SCATTERED = auto()      # [a, b, c, d] - 각 랭크가 자신만
    TP_ATTN_FULL = auto()   # [ab, ab, cd, cd] - TP attn 그룹 내 복제
    FULL = auto()           # [abcd, abcd, abcd, abcd] - 전체 복제

    @staticmethod
    def model_input_output():
        """The scatter mode for model forward pass input and output data"""
        if is_nsa_enable_prefill_cp():
            return ScatterMode.SCATTERED
        return ScatterMode.TP_ATTN_FULL
```

### FlashInfer AllReduce Fusion

```python
# sglang/srt/layers/communicator.py:89-99
FUSE_ALLREDUCE_MAX_BATCH_SIZE = 2048

def apply_flashinfer_allreduce_fusion(batch_size: int):
    return (
        (_is_sm90_supported or _is_sm100_supported)  # Hopper/Blackwell
        and _is_flashinfer_available
        and batch_size > 0
        and batch_size <= FUSE_ALLREDUCE_MAX_BATCH_SIZE
        and not is_dp_attention_enabled()
        and get_global_server_args().enable_flashinfer_allreduce_fusion
    )
```

### LayerCommunicator

SGLang의 핵심 혁신:

```python
# sglang/srt/layers/communicator.py:336-380
class LayerCommunicator:
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
    ):
        self.layer_scatter_modes = layer_scatter_modes
        # 레이어별로 최적의 통신 패턴 결정
        self._communicate_simple_fn = CommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
```

**LayerScatterModes 구조:**
```python
@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode    # 레이어 입력
    attn_mode: ScatterMode           # Attention 계산
    mlp_mode: ScatterMode            # MLP 계산
    middle_residual_mode: ScatterMode # 중간 residual
    layer_output_mode: ScatterMode   # 레이어 출력
```

### DP+TP 이중 병렬화

```python
# sglang/srt/layers/dp_attention.py
def get_attention_tp_group() -> GroupCoordinator:
    return _ATTN_TP_GROUP  # Attention 전용 TP 그룹

def get_attention_dp_size() -> int:
    return _ATTN_DP_SIZE  # DP 크기
```

SGLang은 Attention에 대해 별도의 통신 그룹을 유지하여, DP와 TP를 효율적으로 결합합니다.

### CommunicateContext

```python
# sglang/srt/layers/communicator.py:608-641
@dataclass
class CommunicateContext:
    process_group_sizes: Dict[ScatterMode, int]
    attn_tp_rank: int
    attn_tp_size: int
    attn_dp_size: int
    tp_size: int
    tp_rank: int

    @classmethod
    def init_new(cls):
        process_group_sizes = {
            ScatterMode.SCATTERED: 1,
            ScatterMode.TP_ATTN_FULL: attn_tp_size,
            ScatterMode.FULL: tp_size,
        }
        return cls(...)
```

---

# 4. 구현 패턴 비교

## 4.1 통신 패턴 비교 다이어그램

```
ColumnParallel (f 연산자):
Forward:  X ──[identity]──> X @ A_i ──> Y_i
Backward: dX <──[all-reduce]── dL/dY_i

RowParallel (g 연산자):
Forward:  Y_i ──[all-reduce]──> Y = Σ Y_i
Backward: dY <──[identity]── dL/dY_i
```

## 4.2 f/g 연산자 구현 비교

| 프레임워크 | 구현 방식 | 특징 |
|-----------|----------|------|
| Megatron-LM | `torch.autograd.Function` | 원조, 가장 정교한 비동기 오버랩 |
| nanotron | `Differentiable*` 클래스 | 모듈러, 2가지 TP 모드 |
| DeepSpeed | `ColumnParallel`, `RowParallel` 클래스 | AutoTP와 통합 |
| torchtitan | DTensor 자동 처리 | 선언적, torch.compile 친화 |
| vLLM | `GroupCoordinator` 래퍼 | 추론 최적화, custom op |
| SGLang | `LayerCommunicator` | 레이어별 통신 오케스트레이션 |

## 4.3 프레임워크별 핵심 차이 표

| 측면 | Megatron-LM | nanotron | DeepSpeed | torchtitan | vLLM | SGLang |
|------|-------------|----------|-----------|-----------|------|--------|
| **추상화** | autograd.Function | Differentiable* | AutoTP 주입 | DTensor | GroupCoordinator | LayerCommunicator |
| **통신 오버랩** | CUDA 스트림 | same_device_shard | AsyncColumnParallel | torch.compile | - | FlashInfer 퓨전 |
| **GQA 지원** | 암시적 | contiguous_chunks | SubParam | shape 파라미터 | num_kv_head_replicas | 동일 |
| **고유 최적화** | wgrad 퓨전 커널 | tp_recompute_allgather | tp_grain_size | loss_parallel | custom op | ScatterMode |

## 4.4 MLP 병렬화

모든 프레임워크가 동일한 기본 패턴을 따릅니다:

```
┌─────────────────────────────────────────────────────────────┐
│  Input X [batch, seq, hidden]                               │
│      │                                                      │
│      ▼                                                      │
│  ColumnParallel (fc1/gate_up)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  X @ A_i  (각 랭크가 A의 1/p 열을 담당)              │   │
│  │  통신 없음                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│      │                                                      │
│      ▼ [batch, seq, 4*hidden/p]                            │
│  Activation (GeLU/SiLU)                                    │
│      │                                                      │
│      ▼                                                      │
│  RowParallel (fc2/down)                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Y_i @ B_i  (각 랭크가 B의 1/p 행을 담당)            │   │
│  │  All-Reduce: Z = Σ Y_i @ B_i                        │   │
│  └─────────────────────────────────────────────────────┘   │
│      │                                                      │
│      ▼ [batch, seq, hidden]                                │
│  Output Z (모든 랭크에 복제)                                │
└─────────────────────────────────────────────────────────────┘
```

**차이점:**
- **Megatron-LM:** 비동기 all-reduce로 backward 오버랩
- **nanotron:** `REDUCE_SCATTER` 모드로 Sequence Parallel과 결합 가능
- **DeepSpeed:** `GateUpPack_LinearLayer`로 게이트+업 자동 퓨전
- **torchtitan:** DTensor가 레이아웃 변환 자동 처리
- **vLLM/SGLang:** `gather_output=False`로 중간 통신 스킵

## 4.5 Cross-Entropy 병렬화

학습 프레임워크(Megatron-LM, nanotron)는 vocab-parallel cross-entropy를 구현합니다:

```python
def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    # 각 랭크: [batch*seq, vocab/p] logits 보유

    # 1. 수치 안정성: max logit을 all-reduce (MAX)
    logits_max = vocab_parallel_logits.max(dim=-1)
    all_reduce(logits_max, op=MAX)

    # 2. 타겟 logit: 해당 랭크만 값을 가짐, all-reduce (SUM)
    # 3. 분할 함수: exp(logits).sum()을 all-reduce (SUM)

    # 통신량: O(batch * seq) - vocab 차원 제거됨
```

추론 프레임워크에서는 cross-entropy가 필요 없으므로 구현하지 않습니다.

---

# 5. 성능 고려사항

## 5.1 통신 복잡도

| 연산 | Forward 통신 | Backward 통신 |
|------|------------|--------------|
| ColumnParallel | 없음 | All-Reduce |
| RowParallel | All-Reduce | 없음 |
| VocabParallel Embed | All-Reduce | 없음 |
| VocabParallel CE | 3× All-Reduce | 없음 |

**Transformer 레이어당:** 4× All-Reduce (학습 시 forward + backward)

## 5.2 통신-계산 오버랩

비동기 통신은 다음 조건에서 효과적입니다:
- **CUDA_DEVICE_MAX_CONNECTIONS=1:** 커널 스케줄링 순서 보장
- **큰 hidden dimension:** 계산 시간 > 통신 시간
- **Gradient Accumulation Fusion:** CUDA 커널로 그래디언트 누적

## 5.3 메모리 사용량

| 프레임워크 | 메모리 절약 기법 |
|-----------|----------------|
| Megatron-LM | Activation Checkpointing, Fused Kernels |
| nanotron | `tp_recompute_allgather` 플래그 |
| DeepSpeed | ZeRO-3 + TP 결합 |
| torchtitan | FSDP2 + TP 결합 |
| vLLM | KV Cache 최적화 |
| SGLang | Input Scattered Mode |

---

# 6. 어떤 프레임워크를 선택할 것인가?

## 6.1 학습 목적

| 시나리오 | 추천 프레임워크 | 이유 |
|---------|---------------|------|
| 대규모 사전학습 | **Megatron-LM** | 가장 성숙한 3D 병렬화, 최적화된 커널 |
| 연구/실험 | **nanotron** | 모듈러 설계, 빠른 프로토타이핑 |
| 기존 HF 모델 학습 | **DeepSpeed** | AutoTP, ZeRO 통합 |
| PyTorch 생태계 통합 | **torchtitan** | DTensor 네이티브, torch.compile |

## 6.2 추론 목적

| 시나리오 | 추천 프레임워크 | 이유 |
|---------|---------------|------|
| 일반 LLM 서빙 | **vLLM** | PagedAttention, 성숙한 생태계 |
| 구조화된 출력 | **SGLang** | Grammar 지원, LayerCommunicator |
| 커스텀 최적화 | 둘 다 가능 | 코드베이스 이해 후 확장 |

---

# 7. 결론

Tensor Parallelism의 이론은 2019년 Megatron-LM 논문에서 정립되었지만, 구현은 각 프레임워크의 목적에 따라 크게 달라집니다.

**핵심 교훈:**
1. **f/g 켤레 연산자**는 모든 구현의 기초이지만, DTensor처럼 추상화할 수도 있습니다.
2. **학습 vs 추론**은 최적화 방향이 다릅니다: 그래디언트 통신 vs weight loading
3. **통신-계산 오버랩**은 학습에서 핵심이지만, 추론에서는 덜 중요합니다.
4. **자동화 수준**은 생산성(DeepSpeed AutoTP)과 제어력(Megatron-LM) 사이의 트레이드오프입니다.

실제 시스템을 구축할 때는 이론뿐 아니라 각 프레임워크의 구현 세부사항을 이해하는 것이 중요합니다. 이 글이 그 이해에 도움이 되길 바랍니다.

---

# 부록: 핵심 파일 참조

| 프레임워크 | 핵심 파일 | 주요 내용 |
|-----------|----------|----------|
| Megatron-LM | `megatron/core/tensor_parallel/layers.py` | ColumnParallel, RowParallel |
| Megatron-LM | `megatron/core/tensor_parallel/mappings.py` | f/g 연산자 |
| nanotron | `src/nanotron/parallel/tensor_parallel/nn.py` | TP 레이어 정의 |
| nanotron | `src/nanotron/parallel/tensor_parallel/functional.py` | 비동기 통신 구현 |
| DeepSpeed | `deepspeed/module_inject/layers.py` | Linear 레이어 래퍼 |
| DeepSpeed | `deepspeed/module_inject/auto_tp.py` | AutoTP 파서 |
| DeepSpeed | `deepspeed/module_inject/fusedqkv_utils.py` | Fused QKV 포맷 핸들링 |
| DeepSpeed | `deepspeed/module_inject/tp_shard.py` | 불균등 분할 로직 |
| torchtitan | `torchtitan/distributed/tensor_parallel.py` | DTensor 기반 TP |
| torchtitan | `torchtitan/models/llama3/infra/parallelize.py` | 병렬화 계획 |
| vLLM | `vllm/distributed/parallel_state.py` | GroupCoordinator |
| vLLM | `vllm/model_executor/layers/linear.py` | TP Linear 레이어 |
| SGLang | `sglang/srt/layers/linear.py` | TP Linear 레이어 |
| SGLang | `sglang/srt/layers/communicator.py` | LayerCommunicator |
