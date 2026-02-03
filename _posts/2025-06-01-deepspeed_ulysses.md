---
title: "DeepSpeed Ulysses"
date: 2025-06-01
last_modified_at: 2025-06-01
layout: post
permalink: /blog/deepspeed_ulysses/
description: "All-to-all 통신을 통한 효율적 시퀀스 병렬화로 100만 토큰 이상의 학습을 가능케 하는 DeepSpeed-Ulysses."
tags: sequence-parallelism distributed-training
thumbnail: assets/img/blog/deepspeed-ulysses.png
series: sequence-parallelism
series_order: 3
series_title: "Sequence Parallelism Series"
related_posts: true
disqus_comments: false
giscus_comments: true
toc:
  sidebar: left
---

# TL;DR

> DeepSpeed-Ulysses는 매우 긴 시퀀스(최대 100만 토큰 이상)로 Transformer 모델을 학습하기 위한 시퀀스 병렬화 방법을 제안하며, 입력을 시퀀스 차원으로 분할하고 all-to-all collective 통신을 사용하여 Q, K, V 텐서를 attention head들에 걸쳐 재분배합니다.
> 핵심 혁신은 통신량이 $O(N)$이 아닌 $O(N/P)$로 스케일링되어, 시퀀스 길이 $N$과 GPU 개수 $P$가 비례적으로 증가할 때 통신량이 일정하게 유지된다는 것입니다.
> GPT 모델(최대 300억 파라미터)에 대한 실험 결과, Megatron-LM 시퀀스 병렬화 대비 2.5배 처리량 향상과 4배 긴 시퀀스 지원을 달성했습니다.
> 주요 제약은 시퀀스 병렬화 정도가 attention head 개수에 의해 제한된다는 점이며, 각 GPU는 최소한 하나의 완전한 head를 받아야 합니다.

- Paper Link: [https://arxiv.org/pdf/2309.14509](https://arxiv.org/pdf/2309.14509)

---

# Related Papers

- [Ring Self-Attention](/blog/ring-self-attention/) - 시퀀스 병렬처리에 대한 종합적 관점
- [Blockwise RingAttention](/blog/blockwise_ringattention/) - 링 토폴로지를 활용한 시퀀스 병렬화
- [Unlocking Agentic RL Training for GPT-OSS](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl) - GPT-OSS를 위한 에이전틱 RL 학습
- [DistCA: Efficient Long-Context Language Model Training by Core Attention Disaggregation](https://arxiv.org/pdf/2510.18121) - 코어 어텐션 분리를 통한 효율적 긴 컨텍스트 학습
- [Demystifying NCCL](https://arxiv.org/pdf/2507.04786) - NCCL 통신 라이브러리 심층 분석

---

# Takeaways

## 1. Contribution

대규모 언어 모델의 학습은 네 가지 계산 차원으로 분해될 수 있습니다: 배치 크기(데이터 병렬화로 해결), 은닉 차원(텐서 병렬화로 해결), 레이어 개수(파이프라인 병렬화로 해결), 그리고 시퀀스 길이입니다. 처음 세 차원은 분산 학습 커뮤니티에서 광범위하게 연구되고 최적화되어 온 반면, 시퀀스 차원은 시스템 관점에서 상대적으로 적은 관심을 받았습니다. 실제 응용들이 점점 더 긴 컨텍스트 윈도우를 요구함에 따라 이 격차는 점점 더 중요해지고 있습니다 -- 멀티턴 채팅 히스토리, 책 길이 문서 요약부터 수십억 염기쌍을 가진 유전체 시퀀스와 고차원 시공간 데이터를 가진 기후 모델링까지.

DeepSpeed-Ulysses 이전에는 시퀀스 병렬화를 위한 두 가지 주요 접근법이 존재했습니다. 첫 번째는 Li et al. (2022)이 제안한 Ring Self-Attention (ColAI-SP)으로, query를 로컬로 분할하면서 key와 value를 GPU들에 걸쳐 링 방식으로 전송합니다. 이 접근법은 참여하는 GPU 수와 관계없이 메시지 크기 $M$에 선형적인 통신량을 발생시켜 확장성을 근본적으로 제한합니다. 두 번째 접근법은 Korthikanti et al. (2022)의 Megatron-LM 시퀀스 병렬화로, Megatron의 텐서 병렬화와 긴밀하게 통합되어 allgather 및 reduce-scatter collective를 사용하여 QKV projection을 집계합니다. 효과적이지만, 이 접근법도 더 많은 GPU가 추가될 때 감소하지 않는 $O(N)$ 통신 복잡도를 겪으며, Megatron의 텐서 병렬화 인프라와의 긴밀한 결합이 필요하여 이식성이 제한됩니다. 또한 ColAI-SP는 특정 링 기반 attention 구현을 요구하므로 sparse attention이나 FlashAttention과 같은 다른 attention 메커니즘에 얼마나 잘 일반화되는지 불분명합니다.

DeepSpeed-Ulysses는 개념적으로 간단하지만 매우 효과적인 설계로 이러한 제약들을 해결합니다. 핵심 통찰은 all-to-all collective 통신을 사용함으로써 -- 각 GPU가 모든 head의 시퀀스 분할 청크를 보유하는 것에서 head 부분집합의 전체 시퀀스를 보유하는 것으로 데이터를 재분배 -- 링크당 통신량이 $M$이 아닌 $M/P$가 된다는 것입니다. 이는 시퀀스 길이와 GPU 개수가 모두 비례적으로 스케일링될 때, 장치당 통신 비용이 일정하게 유지되어 통신 구성 요소에 대해 진정한 weak-scaling 동작을 달성함을 의미합니다. 이는 기존 방법 대비 $P$배 감소를 의미하며, 실제 구성에서 10배 이상의 통신 감소로 변환됩니다.

이 기여의 실용적 의의는 여러 차원에 걸쳐 확장됩니다. 첫째, DeepSpeed-Ulysses는 12억 파라미터 모델에서 최대 100만 토큰의 시퀀스 길이로 학습을 가능하게 하며, 사용 가능한 GPU에 따라 선형적으로 스케일링됩니다. 둘째, GPU당 175 TFLOPs 이상의 지속적인 처리량(A100 하드웨어 피크의 54% 이상)을 달성하여 통신 오버헤드가 계산을 병목화하지 않음을 보여줍니다. 셋째, 시스템은 attention-agnostic합니다 -- dense attention, sparse attention, causal attention, cross-attention, 그리고 FlashAttention v2와 같은 효율적인 attention 구현을 수정 없이 지원합니다. 이는 all-to-all 통신 경계가 시퀀스 분할을 attention 계산으로부터 깔끔하게 분리하기 때문입니다. 넷째, ZeRO-3와의 통합은 모델 상태를 시퀀스 병렬 및 데이터 병렬 그룹 모두에 걸쳐 분할할 수 있게 하여, 시퀀스 길이와 모델 크기의 동시 스케일링을 가능하게 합니다. 다섯째, 구현은 이식 가능하고 채택하기 쉬우며, Megatron-LM의 긴밀한 아키텍처 결합과 달리 기존 학습 프레임워크에 최소한의 코드 변경만 필요합니다.

이 기여의 시기는 중요합니다. GPT-4의 128K 컨텍스트 윈도우부터 수십억 뉴클레오타이드의 시퀀스를 처리하는 유전체 언어 모델까지 긴 컨텍스트 응용의 출현이 효율적인 긴 시퀀스 학습 인프라에 대한 긴급한 수요를 창출했기 때문입니다. DeepSpeed-Ulysses는 기존 병렬화 전략과 조합 가능하면서도 이 시스템 수준 격차를 채우는 원칙적이고 확장 가능한 솔루션을 제공합니다.

## 2. Methodology

### 2.1 핵심 직관

긴 시퀀스로 Transformer를 학습하는 근본적인 과제는 attention 메커니즘이 각 토큰이 다른 모든 토큰에 attend하도록 요구한다는 것입니다(dense의 경우). 이는 $O(N^2)$ 계산 복잡도와 attention 행렬에 대한 $O(N^2)$ 메모리 요구사항을 만듭니다. $N$이 매우 클 때(예: 수십만에서 수백만 토큰), 이것은 메모리와 계산 모두 측면에서 단일 GPU에서 불가능해집니다. 자연스러운 해결책은 시퀀스를 여러 GPU에 분산하는 것이지만, 문제는 표준 attention과의 수학적 동등성을 유지하면서 이를 효율적으로 수행하는 방법입니다.

DeepSpeed-Ulysses의 핵심 통찰은 Transformer attention의 multi-head 구조에 뿌리를 두고 있습니다. Multi-head attention에서 은닉 차원 $d$는 $h$개의 head로 분할되며, 각각은 차원 $d_h = d/h$를 가집니다. 중요한 것은 attention이 각 head에 대해 독립적으로 계산된다는 것입니다 -- attention 블록 내에서 head 간 상호작용이 없습니다. 이는 각 GPU가 모든 head의 시퀀스 부분집합이 아닌 head 부분집합의 완전한 시퀀스를 보유하도록 배열할 수 있다면, 각 GPU가 attention 계산 자체 동안 추가 통신 없이 할당된 head들에 대해 완전하고 정확한 attention을 계산할 수 있음을 의미합니다.

All-to-all collective는 이 변환에 이상적인 통신 primitive입니다. Attention 전에 각 GPU는 모든 $h$ head를 가진 $N/P$ 토큰을 보유합니다(시퀀스 분할). All-to-all은 이를 재분배하여 각 GPU가 모든 $N$ 토큰을 보유하지만 $h/P$ head만 보유하도록 합니다(head 분할). Attention 후 또 다른 all-to-all이 이 변환을 역으로 수행하여, 후속 레이어(MLP, layer norm)에 필요한 시퀀스 분할 레이아웃으로 돌아갑니다. 이 이중 all-to-all 패턴은 각 링크가 집계 메시지 크기 $M$에 대해 $M/P$ 데이터만 전송하는 중요한 속성을 가지며, 이는 각 링크가 $P$와 관계없이 $M$을 전송하는 allgather/reduce-scatter 패턴과 대조됩니다.

이 접근법의 이론적 우아함은 병렬화 관심사(GPU에 걸쳐 작업 분산)를 attention 관심사(attention 계산)로부터 깔끔하게 분리하여, 어떤 attention 구현도 두 all-to-all 연산 사이의 블랙박스 로컬 계산으로 사용될 수 있게 한다는 것입니다.

### 2.2 모델 아키텍처

DeepSpeed-Ulysses는 수정 없이 표준 multi-head attention Transformer 아키텍처에서 작동합니다. 시스템 설계는 두 개의 all-to-all 통신 경계로 기존 Transformer 블록을 감쌉니다.

단일 Transformer 레이어를 통한 전체 데이터 흐름은 다음과 같이 진행됩니다:

```
Input: x [N/P, d] on each of P GPUs (sequence-partitioned)
        |
   QKV Projection (local matmul with W_Q, W_K, W_V)
        |
   Q, K, V each [N/P, d] on each GPU
        |
   === All-to-All #1 ===
   Redistribute: [N/P, h, d_h] -> [N, h/P, d_h]
   Each GPU: full sequence, subset of heads
        |
   Local Attention Computation (any implementation)
   Output_context = Softmax(Q K^T / sqrt(d_h)) V
   Result: [N, h/P, d_h] on each GPU
        |
   === All-to-All #2 ===
   Redistribute: [N, h/P, d_h] -> [N/P, h, d_h]
   Each GPU: back to sequence-partitioned
        |
   Output Projection (local matmul with W_O)
        |
   MLP / LayerNorm (all local, sequence-partitioned)
        |
Output: [N/P, d] on each GPU
```

ZeRO-3와의 통합은 데이터 병렬 및 시퀀스 병렬 rank 모두에 걸쳐 모델 상태 분할을 확장합니다. 모든 GPU에서 모델 파라미터 $W_Q, W_K, W_V, W_O$ 및 MLP 가중치를 복제하는 대신, ZeRO-3는 이들을 결합된 데이터 병렬 및 시퀀스 병렬 그룹에 걸쳐 분할합니다. 파라미터가 계산에 필요할 때, 참여 rank로부터 수집(allgathered)되고, gradient는 backward pass 동안 두 그룹에 걸쳐 reduce됩니다. 이 이중 분할은 모델 크기와 시퀀스 길이의 동시 스케일링을 가능하게 합니다.

### 2.3 핵심 알고리즘 및 메커니즘

**Attention을 위한 All-to-All 통신**

중심적인 알고리즘 기여는 시퀀스 분할과 head 분할 텐서 레이아웃 간의 변환을 위해 all-to-all collective 통신을 사용하는 것입니다. $P = 4$ GPU, 시퀀스 길이 $N$, 그리고 head 차원 $d_h$를 가진 $h = 4$ attention head를 가진 구체적인 예를 고려해봅시다.

첫 번째 all-to-all 전에, 각 GPU $i$는 로컬 시퀀스 청크에 대한 QKV projection을 보유합니다: $Q_i, K_i, V_i \in \mathbb{R}^{N/4 \times 4 \times d_h}$. All-to-all 연산은 GPU $i$가 모든 GPU로부터 $i$-번째 head를 받도록 이러한 텐서들을 재분배합니다. All-to-all 후, GPU $i$는 $Q^{(i)}, K^{(i)}, V^{(i)} \in \mathbb{R}^{N \times 1 \times d_h}$를 보유하며, 이는 단일 head에 대한 전체 시퀀스를 나타냅니다. 각 GPU는 그런 다음 표준 scaled dot-product attention을 독립적으로 계산합니다:

$$\text{Output}^{(i)} = \text{Softmax}\left(\frac{Q^{(i)} {K^{(i)}}^T}{\sqrt{d_h}}\right) V^{(i)}$$

이 계산은 비분산 multi-head attention에서 발생하는 것과 수학적으로 동일합니다. Attention이 head마다 독립적으로 계산되기 때문입니다. 근사치가 관여되지 않습니다 -- 결과는 정확합니다.

Attention 계산 후, 두 번째 all-to-all이 재분배를 역으로 수행합니다: 각 GPU는 자신의 head에 대해 계산된 attention 출력을 시퀀스 소유 GPU로 다시 전송하여 $[N/P, h, d_h]$ 레이아웃을 복원합니다. 출력 projection $W_O$ 및 후속 MLP/LayerNorm 연산은 모두 통신 없이 시퀀스 분할 레이아웃에서 작동합니다.

**통신량 분석**

은닉 크기 $h$, 시퀀스 길이 $N$, 그리고 병렬화 정도 $P$를 가진 Transformer에 대해, DeepSpeed-Ulysses는 다음을 수행합니다:
- 집계 메시지 크기 $3Nh$를 가진 QKV projection에 대한 하나의 all-to-all (크기 $N \times h$의 세 행렬)
- 집계 메시지 크기 $Nh$를 가진 출력 context projection에 대한 하나의 all-to-all

노드 내 NVSwitch와 노드 간 fat-tree InfiniBand 토폴로지를 가진 최신 클러스터에서, $P$ GPU에 걸친 집계 메시지 크기 $M$을 가진 all-to-all에 대한 링크당 통신량은 $M/P$입니다. 따라서 DeepSpeed-Ulysses는 다음의 링크당 통신량을 발생시킵니다:

$$V_{\text{DS-Ulysses}} = \frac{4Nh}{P}$$

이는 $O(N/P)$ 복잡도를 가지며, $N$과 $P$가 비례적으로 스케일링될 때 일정합니다.

대조적으로, Megatron-LM 시퀀스 병렬화는 Transformer 레이어당 두 개의 allgather 연산(각각 메시지 볼륨 $Nh$)과 두 개의 reduce-scatter 연산(각각 볼륨 $Nh$)을 수행합니다. Allgather 및 reduce-scatter의 경우, 링크당 볼륨은 $P \gg 1$일 때 $M \cdot \frac{P-1}{P} \approx M$입니다. 따라서 Megatron-LM은 다음의 링크당 통신량을 발생시킵니다:

$$V_{\text{Megatron}} \approx 4Nh$$

이는 $O(N)$ 복잡도를 가집니다 -- $P$와 독립적이며 시퀀스 길이에 선형적으로 증가합니다. 비율은 다음과 같습니다:

$$\frac{V_{\text{Megatron}}}{V_{\text{DS-Ulysses}}} = P$$

$P = 64$ GPU의 경우, DeepSpeed-Ulysses는 링크당 통신량에서 64배 감소를 달성합니다. 이 이론적 이점은 실제로 관찰된 10배 이상의 통신 감소 및 2.5배 처리량 향상으로 직접 변환되며, 이론과 측정된 개선 간의 격차는 학습 시간의 다른 구성 요소(계산, 메모리 액세스) 및 실제 네트워크 효과 때문입니다.

**메모리 효율성을 위한 ZeRO-3 통합**

DeepSpeed-Ulysses는 ZeRO-3의 모델 상태 분할을 데이터 병렬 및 시퀀스 병렬 그룹의 합집합에 걸쳐 확장합니다. $D$ 데이터 병렬 rank와 $S$ 시퀀스 병렬 rank를 가진 구성(총 $D \times S$ GPU)에서, 모델 상태(파라미터, gradient, optimizer 상태)는 모든 $D \times S$ rank에 걸쳐 분할됩니다. 이는 복제된 학습 대비 GPU당 모델 상태 메모리에서 $D \times S$배 감소를 제공합니다.

시퀀스 병렬화와 ZeRO-3 간의 상호작용은 신중한 조율이 필요합니다. Forward pass 동안, 레이어의 파라미터가 필요할 때, 결합된 그룹에 걸쳐 allgathered됩니다. 해당 레이어에 대한 backward pass 후, gradient는 동일한 결합된 그룹에 걸쳐 reduce-scattered됩니다. 시퀀스와 데이터 병렬화 모두에 대한 결합된 그룹의 이중 사용이 DeepSpeed-Ulysses의 메모리 효율성을 Megatron-LM과 구분짓습니다. Megatron-LM은 시퀀스 병렬화를 텐서 병렬화와 결합하며 시퀀스 차원에 걸친 ZeRO의 모델 상태 분할의 이점을 받지 못합니다.

**Attention Agnosticism**

DeepSpeed-Ulysses의 모듈식 설계 -- all-to-all 통신이 attention 계산 전후에 엄격하게 발생 -- 는 로컬 attention이 어떤 구현이든 될 수 있음을 의미합니다. 시스템은 dense self-attention, causal attention(autoregressive 모델에 사용), cross-attention(encoder-decoder 모델에 사용), blocked sparse attention, 그리고 FlashAttention v2와 같은 효율적인 커널을 지원합니다. 이는 특정 링 기반 attention 구현을 요구하고 모든 attention 유형에 얼마나 잘 일반화되는지 불분명한 ColAI-SP(Ring Self-Attention)에 비해 상당한 장점입니다. 유일한 제약은 attention head 개수 $h$가 시퀀스 병렬화 정도 $P$로 나누어떨어져야 한다는 것입니다. Head가 GPU에 걸쳐 균등하게 분배되기 때문입니다.

### 2.4 구현 세부사항

**제약: $P \leq h$**. 시퀀스 병렬화 정도 $P$는 attention head 개수 $h$를 초과할 수 없습니다. 각 GPU는 최소한 하나의 head를 받아야 하기 때문입니다. Key-value head가 query head보다 적은 grouped query attention (GQA)을 가진 모델의 경우, 이 제약은 실제로 더 제한적일 수 있습니다.

**하드웨어 설정**. 모든 실험은 노드 내 NVSwitch 인터커넥트와 노드 간 InfiniBand 연결을 가진 NVIDIA A100 GPU를 사용합니다. 평가는 8개에서 256개 A100 GPU로 스케일링됩니다.

**모델 구성**. 세 가지 GPT 모델 크기가 테스트됩니다: 12억, 70억, 그리고 300억 파라미터. 70억 모델의 경우 32개 GPU가 사용되고, 300억 모델의 경우 64개 GPU가 사용됩니다. ZeRO 병렬화 정도는 GPU 개수와 일치합니다(각각 32와 64).

**통신 복잡도**. 각 Transformer 레이어는 forward pass에서 정확히 두 개의 all-to-all 연산과 backward pass에서 두 개(gradient 흐름을 위해)를 필요로 합니다. 링크당 all-to-all 복잡도는 $O(M/P)$이며, 여기서 $M$은 집계 메시지 크기입니다. 따라서 레이어당 forward 통신은 링크당 $\frac{4Nh}{P}$입니다. $N = 131072$, $h = 4096$ (70억 모델), 그리고 $P = 32$의 경우, 이는 레이어당 링크당 약 64MB로 평가되며 -- NVSwitch 링크의 대역폭 용량 내에 충분히 있습니다.

**메모리 풋프린트**. GPU당 activation 메모리는 시퀀스 분할 부분에 대해 $O(N/P)$로 스케일링되고 attention 계산에 대해 $O(N \cdot h/P)$로 스케일링됩니다(각 GPU는 전체 시퀀스에서 $h/P$ head에 대해서만 attention을 계산하기 때문). 모델 상태 메모리는 ZeRO-3 통합을 통해 결합된 $D \times S$ 분할 계수만큼 감소합니다.

**달성된 처리량**. GPU당 175 TFLOPs 이상의 최대 지속 처리량(BF16/FP16에 대한 A100의 이론적 312 TFLOP 피크의 54% 이상)이 달성되어, 통신 오버헤드가 계산에 의해 잘 숨겨짐을 나타냅니다.

## 3. Results

**시퀀스 길이 확장성**. 첫 번째 평가는 12억 파라미터 GPT 모델에서 시퀀스 길이의 강한 스케일링을 보여줍니다. DeepSpeed-Ulysses는 8개 GPU에서 8K 토큰부터 64개 GPU에서 100만 토큰까지 선형적으로 스케일링하며, 전체적으로 약 100 TFLOPs per GPU를 유지합니다. 이 선형 스케일링은 $O(N/P)$ 통신 속성의 직접적인 결과입니다: $N$이 두 배가 되고 $P$가 두 배가 될 때, GPU당 통신 및 계산은 일정하게 유지됩니다(로컬로 처리되는 attention의 이차 계산 증가는 제외). 테스트된 모든 구성에 걸쳐 지속적으로 높은 처리량은 all-to-all 통신이 가장 큰 규모에서도 병목이 되지 않음을 검증합니다.

**Dense Attention 비교 (70억 모델, 32 GPU)**. DeepSpeed-Ulysses는 70억 dense 모델에서 모든 시퀀스 길이에 걸쳐 Megatron-LM을 지속적으로 능가합니다. 32K 시퀀스 길이에서 DeepSpeed-Ulysses는 약 175 TFLOPs를 달성하는 반면 Megatron-LM은 약 150 TFLOPs를 달성하여 17% 개선을 보입니다. 더 중요하게, Megatron-LM은 128K 및 256K 시퀀스 길이에서 메모리 부족(OOM)으로 실행되는 반면, DeepSpeed-Ulysses는 계속 효율적으로 작동합니다. 64K 토큰에서 DeepSpeed-Ulysses는 약 150 TFLOPs를 달성하는 반면 Megatron-LM은 약 100 TFLOPs로 떨어집니다. 성능 이점은 두 가지 원천에서 비롯됩니다: ZeRO-3 통합으로 DeepSpeed-Ulysses가 더 많은 마이크로 배치 샘플을 맞출 수 있어 GPU 활용도를 증가시키고, all-to-all 통신이 이 워크로드에 대해 allgather/reduce-scatter보다 근본적으로 더 효율적입니다.

**Dense Attention 비교 (300억 모델, 64 GPU)**. 동일한 패턴이 더 큰 규모에서 유지됩니다. DeepSpeed-Ulysses는 8K 토큰에서 약 160 TFLOPs를 달성하고 64K 토큰에서 100 TFLOPs 이상을 유지하는 반면, Megatron-LM은 8K에서 약 80 TFLOPs로 피크를 찍고 64K에서 약 60 TFLOPs로 떨어집니다. Megatron-LM은 128K 및 256K에서 OOM이 되는 반면, DeepSpeed-Ulysses는 이러한 길이를 처리합니다. 2배 처리량 이점은 이론적 $P$배 통신 감소와 일치하며, $P = 64$에서 이론적 개선의 일부라도 상당한 실제 이득으로 변환되기 때문입니다.

**Sparse Attention 비교**. Blocked sparse attention을 사용한 결과는 유사하거나 더 큰 이점을 보여줍니다. 70억 sparse 모델(32 GPU)에서 DeepSpeed-Ulysses는 Megatron-LM의 2배 이상 처리량을 달성합니다. 32K 토큰에서 DeepSpeed-Ulysses는 약 100 TFLOPs에 도달하는 반면 Megatron-LM은 60 TFLOPs입니다. 주목할 만하게, 저자들은 DeepSpeed-Ulysses의 처리량이 통신 프레임워크가 아닌 로컬 sparse attention 구현의 병목 때문에 더 긴 시퀀스에서 감소함을 관찰하여 추가 최적화 잠재력을 시사합니다. 300억 sparse 모델(64 GPU)에서 격차는 더욱 벌어져, DeepSpeed-Ulysses는 32K에서 약 140 TFLOPs를 달성하는 반면 Megatron-LM은 60 TFLOPs입니다.

| 구성 | DS-Ulysses TFLOPs | Megatron TFLOPs | 속도향상 | 최대 시퀀스 길이 |
|---|---|---|---|---|
| 70억 Dense, 32 GPU, 32K | ~175 | ~150 | 1.17x | DS: 256K, Meg: 64K |
| 70억 Dense, 32 GPU, 64K | ~150 | ~100 | 1.5x | - |
| 300억 Dense, 64 GPU, 8K | ~160 | ~80 | 2.0x | DS: 256K, Meg: 64K |
| 300억 Dense, 64 GPU, 64K | ~105 | ~60 | 1.75x | - |
| 70억 Sparse, 32 GPU, 32K | ~100 | ~60 | 1.67x | DS: 128K, Meg: 64K |
| 300억 Sparse, 64 GPU, 32K | ~140 | ~60 | 2.33x | DS: 256K, Meg: 64K |

**병렬 스케일링 연구**. 전역 배치 크기 8을 가진 GPT-70억 dense 모델에서 두 가지 스케일링 연구가 수행됩니다. Strong scaling 연구(고정된 시퀀스 길이 131,072 토큰, GPU 개수 변화)에서 반복 시간은 64 GPU에서 32,432 ms에서 256 GPU에서 9,887 ms로 거의 선형적으로 감소합니다(4배 GPU에 대해 3.28배 속도향상). GPU당 TFLOPs는 165.5에서 136.1로 감소하여, 더 높은 병렬화 정도에서 증가된 통신 오버헤드를 반영합니다 -- 그러나 이는 4배 스케일링 계수에 대해 18%의 적당한 효율성 손실로, 좋은 병렬 효율성을 나타냅니다. Weak scaling 연구(시퀀스 길이와 GPU의 비례 증가)에서, 65K/64 GPU에서 262K/256 GPU로 갈 때, 처리량은 GPU당 161.4에서 147.4 TFLOPs로 감소합니다. Attention 계산이 시퀀스 길이에 이차적이라는 점을 고려하면, 이 9%의 효율성 손실은 놀랍게도 작으며 all-to-all 접근법의 일정 통신 속성을 검증합니다.

**수렴 연구**. 시퀀스 병렬화 정도 4를 가진 8개 A100 GPU에서 32K 시퀀스 길이의 13억 GPT 모델에 대한 수렴 실험은 DeepSpeed-Ulysses가 모델 품질에 영향을 미치지 않는 순수한 시스템 수준 최적화임을 확인합니다. DeepSpeed-Ulysses(ZeRO-1, ZeRO-2, ZeRO-3 포함) 및 Megatron-LM의 손실 곡선은 구별할 수 없으며, 모두 1000 반복 후 약 6.0의 동일한 손실 값으로 수렴합니다. 이는 all-to-all 재분배가 수학적 동등성을 보존하기 때문에 예상됩니다 -- 각 head는 여전히 전체 시퀀스에 걸쳐 정확한 attention을 계산합니다.

실험 결과는 주장된 기여를 포괄적으로 지원합니다. 통신 감소는 이론적 분석($P$배 개선)과 경험적 처리량 이득(최대 2.5배) 모두에 의해 검증됩니다. 4배 더 긴 시퀀스 지원은 여러 모델 크기와 attention 유형에 걸쳐 입증됩니다. Attention agnosticism은 dense 및 sparse attention 모두에서 일관된 결과로 증명됩니다. ZeRO-3 통합은 Megatron-LM에서 OOM이 되는 더 긴 시퀀스를 가능하게 하는 메모리 절감으로 검증됩니다. 수렴 연구는 시스템 최적화가 학습 역학을 손상시키지 않음을 확인합니다.

## 4. Critical Assessment

### Strengths
1. All-to-all 통신 접근법은 allgather/reduce-scatter 대비 링크당 통신량에서 근본적인 $P$배 감소를 달성하며, $N$과 $P$를 비례적으로 스케일링할 때 통신이 일정하다는 깔끔한 이론적 정당화를 가집니다.
2. 설계는 attention-agnostic하며, all-to-all 경계가 통신을 계산으로부터 깔끔하게 분리하기 때문에 dense, sparse, causal, cross-attention, 그리고 FlashAttention v2와 같은 효율적인 구현을 수정 없이 지원합니다.
3. ZeRO-3와의 통합은 모델 크기와 시퀀스 길이의 동시 스케일링을 가능하게 하여, 긴 시퀀스에서 대형 모델을 학습하는 실용적 요구를 해결합니다.
4. 여러 모델 크기(12억, 70억, 300억), attention 유형(dense, sparse), 그리고 병렬화 구성(8-256 GPU)에 걸친 처리량 및 수렴 연구를 포함한 포괄적인 평가.
5. 시스템은 이식 가능하고 통합하기 쉬우며, Megatron-LM의 텐서 병렬화와의 긴밀한 결합과 달리 기존 프레임워크에 최소한의 코드 변경만 필요합니다.

### Limitations
1. 시퀀스 병렬화 정도는 attention head 개수에 의해 제한됩니다($P \leq h$). 이는 더 적은 head를 가진 모델이나 key-value head가 8개 정도로 적을 수 있는 grouped query attention (GQA) 아키텍처에 대한 확장성을 제한합니다.
2. 모든 실험은 NVSwitch를 가진 A100 GPU에서 수행됩니다. 다른 인터커넥트 토폴로지(예: PCIe 기반 시스템, 이기종 클러스터)에서의 all-to-all 통신 성능은 평가되지 않았습니다.
3. 논문은 DeepSpeed-Ulysses가 텐서 병렬화와 어떻게 조합되는지 논의하지 않습니다. 텐서 병렬화도 head 차원에서 작동하므로, 둘을 결합할 때 충돌이 있을 수 있습니다.
4. 수렴 연구는 작은 모델(13억)과 짧은 학습(1000 반복)으로 제한됩니다. 더 큰 모델과 더 긴 학습 실행에 대한 더 광범위한 수렴 검증이 확신을 강화할 것입니다.
5. 논문은 attention 자체의 이차 계산 비용을 해결하지 않습니다 -- 통신 오버헤드만 해결합니다. 극도로 긴 시퀀스의 경우, 로컬 attention 계산은 head당 $O(N^2)$로 남아 있으며, 논문은 이를 관리하기 위해 FlashAttention이나 sparse attention에 의존하지만 해당 문제에 대한 기여는 없습니다.
6. 실험 섹션에서 Ring Self-Attention (ColAI-SP)과의 비교가 없으며, 이론적 비교만 있습니다. 직접적인 경험적 비교가 주장을 강화할 것입니다.
7. 논문은 시퀀스 병렬 그룹에서 GPU 장애에 대한 내결함성이나 복원력을 논의하지 않으며, 이는 대규모 프로덕션 배포에 중요합니다.

### Future Directions
1. Key-value head가 query head에 걸쳐 공유되는 GQA 아키텍처를 지원하도록 접근법을 확장하여, $P$가 query head가 아닌 KV head의 배수가 되도록 할 가능성.
2. DeepSpeed-Ulysses를 텐서 병렬화 및 파이프라인 병렬화와 결합하여 네 가지 차원(배치, 은닉, 깊이, 시퀀스)을 모두 동시에 스케일링하는 "4D 병렬화".
3. 이기종 인터커넥트에서 성능 평가 및 다른 하드웨어 구성에 적응하는 토폴로지 인식 all-to-all 구현 개발.
4. 시퀀스 길이가 점진적으로 증가하는 지속적 사전학습 시나리오를 위해 컨텍스트 길이 확장 기법(예: RoPE interpolation, YaRN)과 통합.
5. 극단적인 시퀀스 길이에서 메모리-계산 트레이드오프를 더욱 최적화하기 위해 시퀀스 병렬화와 activation checkpointing/recomputation 전략 간의 상호작용 탐색.

---
