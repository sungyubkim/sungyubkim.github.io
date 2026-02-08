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

### f 연산자: Copy to TP Region

```python
# megatron/core/tensor_parallel/mappings.py
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

### 비동기 통신 오버랩

Megatron-LM의 핵심 혁신 중 하나는 **통신과 계산의 오버랩**입니다:

```python
# megatron/core/tensor_parallel/layers.py
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        # 1. 입력 그래디언트 계산
        grad_input = grad_output.matmul(weight)

        # 2. 비동기 all-reduce 시작 (통신)
        handle = torch.distributed.all_reduce(
            grad_input, group=ctx.tp_group, async_op=True
        )

        # 3. 가중치 그래디언트 계산 (계산) - 통신과 동시에!
        grad_weight = grad_output.t().matmul(input)

        # 4. all-reduce 완료 대기
        handle.wait()

        return grad_input, grad_weight, ...
```

**핵심 통찰:** `async_op=True`로 all-reduce를 비동기로 시작하고, 그 동안 가중치 그래디언트를 계산합니다. 이를 위해 `CUDA_DEVICE_MAX_CONNECTIONS=1` 환경 변수가 필요합니다.

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

### 두 가지 TP 모드

nanotron의 가장 큰 특징은 **명시적인 2가지 TP 모드**를 지원한다는 점입니다:

```python
# src/nanotron/parallel/tensor_parallel/enum.py
class TensorParallelLinearMode(Enum):
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
```

**ALL_REDUCE 모드:**
```
Input [replicated] → Linear → Output [replicated via all-reduce]
```

**REDUCE_SCATTER 모드:**
```
Input [sharded on batch] → AllGather → Linear → ReduceScatter → Output [sharded]
```

두 번째 모드는 **Sequence Parallelism**과 결합할 때 유용합니다. 배치/시퀀스 차원으로 이미 분할된 데이터를 처리할 때, all-gather와 reduce-scatter를 사용하면 메모리 효율성이 높아집니다.

### tp_recompute_allgather 플래그

nanotron의 독특한 최적화:

```python
# src/nanotron/parallel/tensor_parallel/nn.py
class TensorParallelColumnLinear(nn.Linear):
    def __init__(self, ..., tp_recompute_allgather=False):
        self.tp_recompute_allgather = tp_recompute_allgather
```

이 플래그가 `True`일 때:
- **Forward:** all-gather 결과를 저장하지 않음 (메모리 절약)
- **Backward:** all-gather를 다시 계산 (계산 비용 증가)

**트레이드오프:** 메모리 사용량 ↓ vs 계산 시간 ↑

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

### AutoTP Parser

```python
# deepspeed/module_inject/auto_tp.py
class AutoTP:
    def tp_parser(self, model):
        """모델을 순회하며 TP 적용 가능한 레이어 탐지"""
        for name, child in model.named_modules():
            if 'gate_up_proj' in name:
                return GateUpPack_LinearLayer(child, self.mp_group)
            elif 'down_proj' in name:
                return LinearAllreduce(child, self.mp_group)
            elif 'out_proj' in name or 'o_proj' in name:
                return LinearAllreduce(child, self.mp_group)
            else:
                return LinearLayer(child, self.mp_group)
```

**장점:** 50+ 모델 아키텍처 지원 (Llama, Mistral, Falcon, Bloom 등)
**단점:** 새로운 아키텍처 추가 시 파서 확장 필요

### 8종 Fused QKV 포맷

다양한 모델이 QKV를 다르게 퓨전합니다:

```python
# deepspeed/module_inject/fusedqkv_utils.py
fused_type_dict = {
    'CodeGenBlock': 'codegentype',   # [q(1),q(2),...,k(1),k(2),...,v(1),v(2),...]
    'BloomBlock': 'bloomtype',        # [q(1),k(1),v(1),q(2),k(2),v(2),...]
    'GLMBlock': 'glmtype',            # [Q,Q,...,K,K,...,V,V,...]
    'Phi3DecoderLayer': 'phi3type',
    # ...
}
```

DeepSpeed는 각 포맷에 맞는 전치/분할 로직을 자동 적용합니다.

### SubParameter: 불균등 파티셔닝

GQA(Grouped-Query Attention)처럼 Q와 KV 헤드 수가 다를 때:

```python
# deepspeed/module_inject/layers.py
class SubParamLinearLayer(TensorParallel_Layer):
    def __init__(self, module, mp_group, shape, partition_dim=0):
        # shape = ((q_size, k_size, v_size), -1) for GQA
        # 각 서브파라미터를 독립적으로 분할
```

### Async 통신 오버랩

Megatron-LM과 유사한 패턴:

```python
class AsyncColumnParallel(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        # 비동기 all-reduce 시작
        handle = dist.all_reduce(grad_input, group=ctx.group, async_op=True)

        # 가중치 그래디언트 계산 (통신과 동시)
        grad_weight = grad_output.t().matmul(input)

        handle.wait()
        return grad_input, grad_weight, ...
```

---

## 2.4 torchtitan: PyTorch Native

**핵심 파일:** `torchtitan/distributed/`

torchtitan은 PyTorch의 **DTensor**와 **DeviceMesh** API를 사용하여 TP를 구현합니다. 커스텀 autograd 함수 없이 선언적으로 병렬화를 정의합니다.

### 선언적 병렬화 계획

```python
# torchtitan/models/llama3/infra/parallelize.py
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput,
)

def apply_tp(model, tp_mesh, loss_parallel=False):
    parallelize_module(
        model,
        tp_mesh,
        {
            # 임베딩: Row parallel
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),

            # 최종 정규화: Sequence parallel
            "norm": SequenceParallel(),

            # 출력: Column parallel
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
            ),
        },
    )
```

**핵심 차이점:** 명시적인 f/g 연산자 대신, DTensor가 레이아웃 변환을 자동 처리합니다.

### 레이아웃 명세

```python
from torch.distributed.tensor import Replicate, Shard

Replicate()   # 전체 텐서를 모든 랭크에 복제
Shard(0)      # 0번 차원으로 분할
Shard(1)      # 1번 차원으로 분할 (hidden dim)
Shard(-1)     # 마지막 차원으로 분할 (vocab dim)
```

### Transformer 블록 병렬화

```python
for transformer_block in model.layers.values():
    layer_plan = {
        # Attention 입력: sharded → replicated 변환
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None, None, None),
            desired_input_layouts=(Replicate(), None, None, None),
        ),

        # QKV 프로젝션: Column parallel
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),

        # 출력 프로젝션: Row parallel (출력은 다시 sharded)
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),

        # FFN
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    }

    parallelize_module(transformer_block, tp_mesh, layer_plan)
```

### torch.compile 호환

DTensor 기반 구현은 `torch.compile`과 자연스럽게 통합됩니다:

```python
# Async TP 활성화
torch._inductor.config._micro_pipeline_tp = True
```

### DeviceMesh 구성

```python
# torchtitan/distributed/parallel_dims.py
@dataclass
class ParallelDims:
    dp_replicate: int  # 데이터 병렬 복제
    dp_shard: int      # 데이터 병렬 샤딩 (FSDP)
    cp: int            # 컨텍스트 병렬
    tp: int            # 텐서 병렬
    pp: int            # 파이프라인 병렬
    ep: int            # 전문가 병렬
```

다차원 메시를 생성하여 복잡한 병렬화 조합을 지원합니다.

---

# 3. 추론 프레임워크

추론 프레임워크는 학습과 다른 최적화 방향을 가집니다:
- **그래디언트 없음:** backward pass 최적화 불필요
- **레이턴시 중심:** 단일 토큰 생성 시간이 중요
- **Weight Loading:** 모델 로딩 시 샤딩 적용

## 3.1 vLLM: 추론 최적화의 정석

**핵심 파일:** `vllm/distributed/`, `vllm/model_executor/layers/`

### GroupCoordinator 추상화

```python
# vllm/distributed/parallel_state.py
class GroupCoordinator:
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_  # 단일 GPU면 통신 스킵

        # torch.compile 호환을 위한 custom op 사용
        return torch.ops.vllm.all_reduce(
            input_, group_name=self.unique_name
        )
```

**핵심:** 단일 GPU 시 통신 스킵, custom op으로 torch.compile 지원

### Weight Loader 패턴

추론에서는 모델 로딩 시 weight를 분할합니다 (런타임 아님):

```python
# vllm/model_executor/layers/linear.py
class ColumnParallelLinear(LinearBase):
    def weight_loader(self, param, loaded_weight):
        output_dim = getattr(param, "output_dim", None)

        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size

            # 로드 시점에 weight를 분할
            loaded_weight = loaded_weight.narrow(
                output_dim, start_idx, shard_size
            )

        param_data.copy_(loaded_weight)
```

**장점:** 런타임에 분할 연산 없음, 추론 레이턴시 최소화

### input_is_parallel 플래그

```python
class RowParallelLinear(LinearBase):
    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_  # 이미 분할되어 있음, scatter 스킵
        else:
            # 입력 분할 필요
            splitted = split_tensor_along_last_dim(input_, self.tp_size)
            input_parallel = splitted[self.tp_rank]
```

ColumnParallel의 출력을 바로 RowParallel에 연결할 때, 중간 scatter를 스킵합니다.

### QKV Parallel: GQA/MQA 지원

```python
# vllm/model_executor/layers/linear.py
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, ..., total_num_kv_heads):
        tp_size = get_tensor_model_parallel_world_size()

        if tp_size >= total_num_kv_heads:
            # KV 헤드 < TP 크기: KV 헤드 복제
            self.num_kv_heads = 1
            self.num_kv_head_replicas = tp_size // total_num_kv_heads
        else:
            # KV 헤드 >= TP 크기: KV 헤드 분할
            self.num_kv_heads = total_num_kv_heads // tp_size
            self.num_kv_head_replicas = 1
```

GQA(Grouped-Query Attention)에서 KV 헤드 수가 Query 헤드보다 적을 때, 자동으로 복제/분할을 결정합니다.

---

## 3.2 SGLang: 통신 오케스트레이션

**핵심 파일:** `sglang/srt/layers/`

SGLang은 vLLM 기반이지만, 구조화된 출력(structured output) 지원과 함께 독자적인 최적화를 추가했습니다.

### LayerCommunicator

SGLang의 핵심 혁신:

```python
# sglang/srt/layers/communicator.py
class LayerCommunicator:
    def __init__(self):
        self.scatter_mode = ScatterMode.SCATTERED  # 또는 FULL, TP_ATTN_FULL
```

**세 가지 scatter 모드:**
- `SCATTERED`: hidden states가 TP 차원으로 분할된 상태
- `FULL`: hidden states가 모든 랭크에 복제된 상태
- `TP_ATTN_FULL`: Attention에서만 복제, 나머지는 분할

LayerCommunicator는 레이어별로 최적의 통신 패턴을 결정합니다.

### Input Scattered Mode

```python
# 디코드 단계에서 통신 최소화
enable_attn_tp_input_scattered = True
```

이 모드가 활성화되면:
- Hidden states는 TP 차원으로 분할된 채로 유지
- QKV 프로젝션 직전에만 all-gather
- 디코드 시 토큰당 통신 오버헤드 감소

### DP+TP 이중 병렬화

```python
# sglang/srt/layers/dp_attention.py
def get_attention_tp_group():
    """Attention 전용 TP 그룹 반환"""
    # DP와 TP를 결합한 특수 그룹
```

SGLang은 Attention에 대해 별도의 통신 그룹을 유지하여, DP와 TP를 효율적으로 결합합니다.

### 다중 통신 백엔드

```python
# sglang/srt/distributed/device_communicators/
- pynccl.py           # PyNCCL (기본)
- custom_all_reduce.py # 링 기반 커스텀 all-reduce
- torch_symm_mem.py   # NVIDIA IPC 기반 zero-copy
- msccl.py            # MSCCL++ 최적화
```

SGLang은 하드웨어와 워크로드에 따라 최적의 백엔드를 선택합니다.

---

# 4. 구현 패턴 비교

## 4.1 f/g 연산자 구현

| 프레임워크 | 구현 방식 | 특징 |
|-----------|----------|------|
| Megatron-LM | `torch.autograd.Function` | 원조, 가장 정교한 비동기 오버랩 |
| nanotron | `Differentiable*` 클래스 | 모듈러, 2가지 TP 모드 |
| DeepSpeed | `ColumnParallel`, `RowParallel` 클래스 | AutoTP와 통합 |
| torchtitan | DTensor 자동 처리 | 선언적, torch.compile 친화 |
| vLLM | `GroupCoordinator` 래퍼 | 추론 최적화, custom op |
| SGLang | `LayerCommunicator` | 레이어별 통신 오케스트레이션 |

## 4.2 MLP 병렬화

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

## 4.3 Cross-Entropy 병렬화

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
| torchtitan | `torchtitan/distributed/tensor_parallel.py` | DTensor 기반 TP |
| torchtitan | `torchtitan/models/llama3/infra/parallelize.py` | 병렬화 계획 |
| vLLM | `vllm/distributed/parallel_state.py` | GroupCoordinator |
| vLLM | `vllm/model_executor/layers/linear.py` | TP Linear 레이어 |
| SGLang | `sglang/srt/layers/linear.py` | TP Linear 레이어 |
| SGLang | `sglang/srt/layers/communicator.py` | LayerCommunicator |
