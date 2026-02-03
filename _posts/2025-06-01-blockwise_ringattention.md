---
title: "Blockwise RingAttention"
date: 2025-06-01
last_modified_at: 2025-06-01
layout: post
permalink: /blog/blockwise_ringattention/
description: "Blockwise RingAttention은 blockwise attention의 순열 불변성을 활용하여 KV 블록 통신과 계산을 완전히 중첩시킴으로써, 장치 수에 선형 비례하는 컨텍스트 길이 확장을 제로 오버헤드로 달성하는 분산 시퀀스 병렬화 방법."
tags: sequence-parallelism ring-attention memory-efficiency distributed-training long-context-transformers memory-efficient-attention distributed-computing
thumbnail: assets/img/blog/blockwise-ringattention.png
series: sequence-parallelism
series_order: 2
series_title: "Sequence Parallelism Series"
related_posts: true
disqus_comments: false
giscus_comments: true
toc:
  sidebar: left
---

# TL;DR

> 이 논문은 Blockwise RingAttention을 제안합니다. 이는 긴 시퀀스를 링 토폴로지로 연결된 여러 장치에 분산시키면서, key-value 블록의 통신과 blockwise attention 계산을 완전히 중첩시켜, 장치 수에 선형적으로 비례하는 컨텍스트 길이를 가능하게 합니다.
> 핵심 혁신은 blockwise attention의 내부 루프가 가진 순열 불변성을 활용하여, KV 블록 전송을 계산과 파이프라인 방식으로 처리하며 오버헤드가 전혀 없도록 한 것입니다.
> TPUv4-1024에서의 실험은 1억 개 이상의 토큰 컨텍스트 크기를 달성했으며(기존 메모리 효율적 방법 대비 최대 512배 향상), MFU 저하는 무시할 만한 수준입니다.
> 이 접근법은 정확한 attention 계산(근사 없음)을 수행하며 기존 병렬화 전략과 조합 가능하지만, 계산-대역폭 비율에 의해 결정되는 최소 블록 크기가 필요합니다.

- Paper Link: [https://openreview.net/forum?id=WsRHpHH4s0](https://openreview.net/forum?id=WsRHpHH4s0)

---

# Related Papers

**분산 어텐션 방법론:**
- [DISTFLASHATTN](https://arxiv.org/pdf/2310.03294) - GPU 간 FlashAttention 분산을 위한 방법
- [Striped Attention](https://arxiv.org/pdf/2311.09431) - 로드 밸런싱을 위한 대안적 어텐션 분배 패턴
- [DeepSpeed Ulysses](https://arxiv.org/pdf/2309.14509) - 어텐션 분산을 활용한 시퀀스 병렬처리

**긴 시퀀스 훈련:**
- [Ring Self-Attention](/blog/ring-self-attention/) - 시퀀스 병렬처리에 대한 종합적 관점
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/pdf/2411.01783) - 추론을 위한 컨텍스트 레벨 병렬처리
- [LoongTrain](https://arxiv.org/pdf/2406.18485) - 매우 긴 시퀀스를 위한 2D 어텐션 병렬처리

**메모리 최적화:**
- [Reducing Activation Recomputation in Large Transformer Models](/blog/sp/) - 메모리 효율적인 훈련 기법
- [USP](/blog/usp/) - Ring과 Ulysses 방법을 결합한 통합 접근법

---

# Takeaways

## 1. Contribution

이 논문이 다루는 핵심 과제는 Transformer 모델이 매우 긴 시퀀스를 처리하는 것을 막는 메모리 병목입니다. 표준 self-attention은 시퀀스 길이 $s$에 대해 $O(s^2)$의 메모리를 필요로 하며, 이는 컨텍스트 윈도우가 커질수록 극복할 수 없는 장벽이 됩니다. FlashAttention이나 Blockwise Parallel Transformers(BPT)와 같은 메모리 효율적 attention 메커니즘이 등장했음에도 불구하고, 근본적인 제약이 여전히 존재합니다. 각 transformer 레이어는 완전한 출력 텐서($b \times s \times h$ 형태, 여기서 $b$는 배치 크기, $s$는 시퀀스 길이, $h$는 은닉 차원)를 저장해야 하는데, 이는 다음 레이어의 self-attention이 모든 위치에 접근해야 하기 때문입니다. 은닉 크기가 1024인 1억 개의 토큰의 경우, 이것만으로도 1000 GB 이상의 메모리가 필요하며, 이는 단일 가속기의 용량(일반적으로 100 GB HBM 미만)을 훨씬 초과합니다. 이것이 RingAttention이 목표로 하는 핵심 병목입니다.

논문의 주요 기여는 장치당 메모리 제약을 완전히 제거하는 분산 시퀀스 병렬화 방식으로, 컨텍스트 크기가 장치 수에 선형적으로 비례하여 확장될 수 있게 합니다. 통신 오버헤드를 완전히 숨길 수 없거나, 각 장치에서 전체 시퀀스를 수집해야 하거나, attention 근사에 의존하는 기존 시퀀스 병렬화 접근법과 달리, RingAttention은 핵심적인 수학적 속성을 활용하여 제로 오버헤드 통신 중첩을 달성합니다. 이 속성은 self-attention의 blockwise 계산이 key-value 블록이 처리되는 순서에 대해 순열 불변이라는 것입니다. 즉, 실행 통계(수치적 안정성을 위한 분자, 분모, 최대 점수)가 올바르게 유지되는 한, 블록을 어떤 순서로 처리하더라도 결과는 표준 attention과 수학적으로 동일합니다.

이 기여의 실용적 의미는 상당합니다. RingAttention 이전에는 사용 가능한 장치가 얼마나 많든 간에 최대 컨텍스트 길이가 단일 장치의 메모리에 의해 제한되었습니다. 기존 최고 수준인 BPT조차 TPUv4-1024에서 7B 모델을 16K 토큰으로만 학습할 수 있었습니다. 동일한 하드웨어에서 RingAttention을 사용하면 컨텍스트가 8.19M 토큰으로 확장되어 512배 향상됩니다. 32개의 A100 GPU에서 7B 모델은 128K에서 4M 토큰(32배)으로 증가합니다. 이 접근법은 정확하며 attention 계산에 근사를 도입하지 않아 모델 품질 유지에 중요합니다. 또한 기존 병렬화 전략(FSDP, 텐서 병렬화)과 조합 가능하며, 드롭인 컴포넌트로 모든 메모리 효율적 로컬 attention 구현(FlashAttention, BPT)을 사용할 수 있습니다.

이 연구의 시의성은 두 가지 트렌드의 수렴에서 비롯됩니다. 빠르게 증가하는 긴 컨텍스트 모델의 필요성(책, 비디오, 코드베이스, 과학 데이터 처리)과 고대역폭 인터커넥트(NVLink, TPU ICI)를 갖춘 멀티 장치 클러스터의 가용성입니다. Li 등(2023b)의 링 기반 접근법과 같은 기존 시퀀스 병렬화 방법은 blockwise 계산을 중심으로 설계되지 않았기 때문에 완전한 통신-계산 중첩을 달성할 수 없었습니다. blockwise transformer와 링 통신 토폴로지의 결합이 RingAttention을 실용적이고 이론적으로 깔끔하게 만드는 요소입니다.

부차적 기여는 여러 축에 걸친 실증적 검증입니다. 다양한 하드웨어(NVLink/InfiniBand가 있는 A100 GPU, TPUv3/v4/v5e)에서의 최대 컨텍스트 길이 벤치마크, 무시할 수 있는 오버헤드를 보여주는 모델 FLOPs 활용률 분석, 더 긴 컨텍스트가 컨텍스트 내 RL 성능을 향상시킴을 보여주는 강화 학습 실험, 그리고 미세 조정된 RingAttention-13B-512K 모델이 GPT-3.5, Vicuna, Claude-2가 실패하는 컨텍스트 길이에서 정확도를 유지함을 보여주는 장거리 검색 평가가 포함됩니다.

## 2. Methodology

### 2.1 핵심 직관

RingAttention의 이론적 기초는 두 가지 기둥에 있습니다. blockwise attention 계산과 softmax 분자-분모 분해의 순열 불변성 속성입니다.

표준 self-attention은 $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$를 계산하며, 여기서 softmax는 행 단위로 적용됩니다. 선행 연구(Milakov and Gimelshein, 2018; Rabe and Staats, 2021)의 핵심 통찰은 softmax가 전체 $s \times s$ attention 행렬을 구체화하지 않고도 점진적으로 계산될 수 있다는 것입니다. 구체적으로, 주어진 쿼리 블록 $Q_i$에 대해, key-value 블록 $(K_j, V_j)$를 한 번에 하나씩 반복하면서 가중합(분자), 정규화 상수(분모), 실행 최대 점수(수치적 안정성을 위한)에 대한 실행 누적기를 유지할 수 있습니다. 모든 key-value 블록을 처리한 후, 분자를 분모로 나누어 최종 출력을 얻습니다. 중요한 것은 블록 간 재조정이 올바르게 수행되는 한, key-value 블록이 처리되는 순서가 최종 결과에 영향을 미치지 않는다는 것입니다. 이것이 순열 불변성 속성입니다.

Blockwise Parallel Transformers(BPT)는 feedforward 네트워크도 blockwise 방식으로 계산하여, 다음 쿼리 블록으로 이동하기 전에 각 블록에 대한 attention 출력과 융합함으로써 이 아이디어를 확장했습니다. 이를 통해 최대 활성화 메모리를 레이어당 $8bsh$(feedforward 중간 값)에서 $2bsh$로 줄였습니다. 그러나 BPT를 사용하더라도, 단일 장치는 여전히 $b \times s \times h$ 형태의 전체 레이어 출력을 저장해야 하는데, 이는 다음 레이어가 attention 계산을 위해 모든 위치를 필요로 하기 때문입니다.

RingAttention의 핵심 통찰은 시퀀스가 $N$개의 장치에 분할될 때, 각 장치는 자체 쿼리 블록만 영구적으로 필요하고, key-value 블록은 파이프라인 방식으로 장치를 순환할 수 있다는 것입니다. 순열 불변성 때문에 KV 블록의 순서는 중요하지 않습니다. 따라서 장치를 링으로 배열할 수 있으며, 각 단계에서 각 장치는 로컬 쿼리 블록과 현재 사용 가능한 KV 블록 사이의 attention을 계산한 다음, 해당 KV 블록을 다음 장치로 전달합니다. 한 블록에 대한 계산 시간이 KV 블록 전송에 대한 통신 시간을 초과하면, 통신이 완전히 중첩되어 추가 오버헤드가 없게 됩니다. 이는 메모리 병목을 단일 장치 제약에서 멀티 장치 제약으로 변환하며, 컨텍스트 길이가 장치 수에 선형적으로 확장됩니다.

### 2.2 모델 아키텍처

RingAttention은 Transformer 아키텍처 자체를 수정하지 않습니다. 원래 Transformer와 동일한 모델을 사용하지만 장치 간 계산을 재구성합니다. 아키텍처는 blockwise parallel transformer 위에 계층화된 분산 실행 패턴으로 설명할 수 있습니다.

시스템은 $N_h$개의 호스트 장치를 논리적 링으로 구성합니다. host-1, host-2, ..., host-$N_h$입니다. 길이 $s$의 입력 시퀀스는 크기 $c = s / N_h$의 $N_h$개의 연속적인 블록으로 분할됩니다. 각 호스트 $i$에는 다음이 할당됩니다.

- 형태 $(b, c, d)$의 로컬 쿼리 블록 $Q_i$
- 형태 $(b, c, d)$의 로컬 키 블록 $K_i$
- 형태 $(b, c, d)$의 로컬 값 블록 $V_i$

실행은 레이어별로 진행됩니다. 각 레이어 내에서:

```
각 transformer 레이어에 대해:
  [외부 루프 - 호스트 간 분산]
  각 호스트 i는 Q_i를 영구적으로 보유

  [내부 루프 - KV 블록이 링을 통해 순환]
  count = 1부터 N_h까지:
    각 호스트가 동시에:
      1. blockwise attention(Q_i, K_current, V_current) 계산
         분자, 분모, max_score를 점진적으로 업데이트
      2. (K_current, V_current)를 host-(i+1)로 전송
         (K_next, V_next)를 host-(i-1)로부터 수신

  [Feedforward - 로컬 계산]
  각 호스트는 자신의 attention 출력에 대해 blockwise FFN 계산
```

중요한 속성은 단계 1과 2가 동시에 발생한다는 것입니다. 호스트가 현재 KV 블록에 대한 attention을 계산하는 동안, 동시에 해당 블록을 다음 호스트로 전송하고 이전 호스트로부터 새 블록을 수신합니다. $N_h$번의 순환 후, 모든 호스트는 모든 KV 블록을 보았으며 올바른 attention 출력을 생성할 수 있습니다.

### 2.3 핵심 알고리즘과 메커니즘

**점진적 Blockwise Attention.** 핵심 계산 기본 요소는 점진적 blockwise attention 계산입니다. 쿼리 블록 $Q_i$와 key-value 블록 시퀀스 $(K_1, V_1), (K_2, V_2), \ldots, (K_{N_h}, V_{N_h})$에 대해, attention 출력은 점진적으로 계산됩니다. 각 단계 $j$에서, 알고리즘은 로컬 attention 점수 $S_{ij} = Q_i K_j^T / \sqrt{d}$를 계산하며, 여기서 $S_{ij} \in \mathbb{R}^{c \times c}$입니다. 그런 다음 세 개의 실행 누적기를 업데이트합니다. 첫째, 수치적 안정성을 위해 필요한 실행 최댓값 $m_j = \max(m_{j-1}, \text{rowmax}(S_{ij}))$입니다. 둘째, softmax 정규화 상수를 누적하는 실행 분모 $\ell_j = e^{m_{j-1} - m_j} \cdot \ell_{j-1} + \text{rowsum}(e^{S_{ij} - m_j})$입니다. 셋째, 가중치 값 합을 누적하는 실행 분자 $O_j = e^{m_{j-1} - m_j} \cdot O_{j-1} + e^{S_{ij} - m_j} \cdot V_j$입니다. 모든 $N_h$ 블록을 처리한 후, 최종 출력은 $\text{output}_i = O_{N_h} / \ell_{N_h}$이며, 이는 쿼리 블록 $i$에 대한 표준 attention 출력과 수학적으로 동일합니다. 이 점진적 계산은 FlashAttention 및 BPT와 동일하며, RingAttention의 기여는 이를 장치 간에 분산시킨 것입니다.

**링 통신 패턴.** 링 통신은 장치의 순열을 따라 동시에 전송 및 수신을 수행하는 집합 연산 `jax.lax.ppermute`를 사용하여 구현됩니다. 각 내부 루프 단계에서, 모든 장치는 현재 KV 블록을 링의 다음 장치로 전송하고(호스트 $i$는 호스트 $i+1 \mod N_h$로 전송) 이전 장치로부터 수신합니다(호스트 $i$는 호스트 $i-1 \mod N_h$로부터 수신). 이것은 링의 인접한 호스트만 관련되는 포인트-투-포인트 통신 패턴이며, GPU all-to-all 토폴로지와 TPU 토러스 토폴로지 모두와 호환됩니다. all-gather 또는 all-reduce 패턴에 비한 주요 이점은 각 장치가 두 개의 이웃과만 통신하여 전체 장치 수와 관계없이 단계당 통신 볼륨을 $4cd$ 바이트(bfloat16의 $c \times d$ 요소의 두 블록)로 일정하게 유지한다는 것입니다.

**통신-계산 중첩 조건.** 제로 오버헤드 작동을 위해서는 계산 시간이 통신 시간을 초과해야 합니다. 한 blockwise attention 단계에 대한 계산은 $4dc^2$ FLOPs가 필요합니다(쿼리-키 내적에 $2dc^2$, 점수-값 곱에 $2dc^2$를 포함). 통신은 $4cd$ 바이트(키 및 값 블록)의 전송이 필요합니다. 따라서 중첩 조건은:

$$\frac{4dc^2}{F} \geq \frac{4cd}{B} \implies c \geq \frac{F}{B}$$

여기서 $F$는 장치의 FLOPS이고 $B$는 장치 간 대역폭입니다. 이는 블록 크기 $c$가 최소한 $F/B$이어야 함을 의미합니다. NVLink가 있는 A100 GPU의 경우($F = 312$ TFLOPS, $B = 300$ GB/s), 최소 블록 크기는 약 1K 토큰입니다. InfiniBand로 연결된 A100의 경우($B = 12.5$ GB/s), 요구 사항은 약 25K 토큰으로 증가합니다. 장치당 총 시퀀스는 최소 $6c$이어야 하므로(6개 블록을 수용하기 위해: 1개의 쿼리, 2개의 현재 KV, 2개의 수신 중인 KV, 1개의 출력), 최소 장치당 시퀀스 길이는 인터커넥트에 따라 6K에서 150K까지 범위입니다.

**메모리 분석.** 각 호스트는 크기 $b \times c \times h$의 정확히 6개의 블록을 저장합니다. 쿼리 블록 1개, 현재 키 및 값 블록 2개, 수신 중인 키 및 값 블록 2개, attention 출력 1개입니다. 이는 레이어당 총 $6bch$ 바이트의 활성화 메모리를 제공합니다. 중요한 것은 이것이 총 시퀀스 길이 $s$와 독립적이며 하드웨어의 계산 대 대역폭 비율에 의해 결정되는 블록 크기 $c$에만 의존한다는 것입니다. 바닐라 Transformer($2bhs^2$), 메모리 효율적 attention($8bsh$), BPT($2bsh$)와 비교하여, RingAttention은 더 작고 시퀀스 길이로부터 분리된 메모리를 달성합니다.

**역전파.** 역전파는 동일한 링 통신 패턴을 사용합니다. 역전파 동안, 각 장치는 실행 그래디언트 누적기 $dQ$, $dK$, $dV$를 유지하고 KV 블록과 그들의 그래디언트를 링을 통해 순환시킵니다. 동일한 중첩 조건이 적용됩니다. 구현은 JAX의 `custom_vjp`를 사용하여 순방향 및 역방향 패스를 명시적으로 정의하여, 링 통신이 자동 미분과 적절하게 통합되도록 합니다.

**인과 마스킹.** 인과 마스킹을 사용하는 자기회귀 모델의 경우, attention 바이어스(마스크)가 블록별로 적용됩니다. 각 장치는 쿼리 블록과 현재 KV 블록의 전역 인덱스를 기반으로 인과 마스크의 적절한 슬라이스를 계산합니다. 구현은 `lax.dynamic_slice_in_dim`을 사용하여 링 순환 중 각 (쿼리 블록, KV 블록) 쌍에 대한 올바른 바이어스 슬라이스를 추출합니다.

### 2.4 구현 세부사항

구현은 JAX로 되어 있으며 링 통신에는 `jax.lax.ppermute`를, KV 블록에 대한 내부 루프에는 `lax.scan`을 사용합니다. 순방향 패스는 분자, 분모, 최대 점수에 대한 누적기를 유지하며, 이들은 각각 0과 음의 무한대로 초기화됩니다. 블록 크기(`query_chunk_size` 및 `key_chunk_size`)는 각 장치 내에서 로컬 blockwise 계산의 세분성을 제어하는 구성 가능한 하이퍼파라미터입니다.

대규모 학습의 경우, 권장 구성은 FSDP(모델 매개변수 샤딩), 선택적 텐서 병렬화(전역 배치 크기 감소용), RingAttention(컨텍스트 길이 확장용)을 결합합니다. 각 병렬화 차원의 정도는 `mesh_dim` 매개변수를 통해 제어됩니다. 예를 들어, 512개의 A100 GPU와 30B 모델의 경우, FSDP로 8개의 장치에 모델을 샤딩하고 나머지 64개의 장치를 RingAttention에 사용하여 64배의 컨텍스트 길이 증가를 얻을 수 있습니다.

중첩 조건에 대한 하드웨어 요구 사항은 논문에 문서화되어 있습니다. 최소 블록 크기는 고대역폭 인터커넥트(NVLink, TPU ICI)의 경우 약 1K 토큰, InfiniBand의 경우 약 25K입니다. 최소 장치당 시퀀스 길이는 블록 크기의 $6 \times$입니다. 모든 실험은 표준 관행에 따라 attention과 feedforward 모두에 대해 전체 그래디언트 체크포인팅을 사용합니다. 학습은 전체 정밀도로 수행됩니다(GPU에서는 float32, TPU에서는 float32 누적을 사용한 bfloat16 matmul).

컨텍스트 크기에 따른 데이터셋당 FLOPs 확장은 다음 공식을 따릅니다:

$$\text{FLOPs ratio} = \frac{6h + s_2}{6h + s_1}$$

여기서 $s_1$과 $s_2$는 이전 및 새로운 컨텍스트 길이이고, $h$는 은닉 차원입니다. 이는 더 큰 모델(더 큰 $h$)의 경우, 더 긴 컨텍스트로부터의 상대적 FLOPs 증가가 더 완만함을 보여줍니다. 175B 모델(은닉 차원 12288)의 경우, 4K에서 1M 컨텍스트로 확장하면 컨텍스트가 256배 길어졌음에도 불구하고 데이터셋당 FLOPs는 14.4배만 증가합니다.

## 3. Results

논문은 RingAttention을 네 가지 차원에서 평가합니다. 최대 컨텍스트 크기, 모델 FLOPs 활용률(MFU), 컨텍스트 내 강화 학습, 장거리 검색입니다.

**최대 컨텍스트 크기.** 가장 놀라운 결과는 컨텍스트 크기 확장 실험입니다(표 3). 테스트된 모든 하드웨어 구성에서 RingAttention은 BPT 기준선 대비 장치 수에 선형적으로 비례하여 확장되는 컨텍스트 길이를 달성합니다. NVLink가 있는 8개의 A100 GPU에서 3B 모델은 64K(BPT)에서 512K(RingAttention)로 8배 향상되며, 이는 8개의 장치와 일치합니다. TPUv4-1024에서 3B 모델은 16.38M 토큰을 달성하고(BPT의 32K 대비 512배), 30B 모델은 2.05M 토큰에 도달합니다(BPT의 8K 대비 256배). 이러한 결과는 컨텍스트가 $N_h \times s_{\text{baseline}}$로 확장된다는 이론적 예측을 확인하며, 여기서 $N_h$는 RingAttention에 전념하는 장치 수입니다. 향상 인수는 RingAttention 장치 수와 같으며, 이는 제로 오버헤드 주장을 검증합니다.

| 하드웨어 | 모델 | BPT 최대 컨텍스트 | RingAttention 최대 컨텍스트 | 향상 |
|----------|-------|----------------|--------------------------|-------------|
| 8x A100 NVLink | 3B | 64K | 512K | 8x |
| 8x A100 NVLink | 7B | 32K | 256K | 8x |
| 32x A100 InfiniBand | 7B | 128K | 4,096K | 32x |
| TPUv4-1024 | 3B | 32K | 16,384K | 512x |
| TPUv4-1024 | 7B | 16K | 8,192K | 512x |
| TPUv4-1024 | 30B | 8K | 2,048K | 256x |

**모델 FLOPs 활용률.** MFU 실험(표 4)은 RingAttention이 무시할 수 있는 오버헤드를 도입함을 보여줍니다. 동일한 구성에서 BPT와 비교할 때, 실제 MFU는 예상 MFU(더 긴 시퀀스에서 self-attention FLOPs의 증가된 비율을 고려하여 계산)를 밀접하게 추적합니다. self-attention은 feedforward 연산보다 낮은 산술 강도를 가지므로, 더 긴 컨텍스트 길이는 자연스럽게 FLOPs 분포를 본질적으로 낮은 MFU를 가진 attention 쪽으로 이동시킵니다. 논문은 실제 RingAttention MFU가 예상 값과 일치함을 보여주며, 이는 링 통신이 실제로 완전히 중첩됨을 확인합니다. 예를 들어, 32개의 A100에서 2M 컨텍스트로 학습하는 13B 모델은 BPT 기준선의 컨텍스트를 단순히 확장하여 예상되는 것과 비교 가능한 MFU를 달성합니다.

**컨텍스트 내 RL 성능.** ExoRL에 대한 강화 학습 실험(표 5)은 다운스트림 작업 검증을 제공합니다. Agentic Transformer(AT) 프레임워크를 사용하여, RingAttention은 128개의 궤적(각 4000 토큰)에 대한 조건화를 가능하게 하며, 이는 총 512K 토큰으로 동일한 하드웨어의 BPT에서는 불가능합니다(128 궤적 구성은 메모리 효율적 attention과 BPT 모두에서 OOM을 발생시킵니다). AT+RingAttention 모델은 총 평균 리턴 113.66을 달성하며, 이는 AT+BPT의 111.13(32 궤적 사용)과 비교됩니다. 이는 컨텍스트 내에서 더 많은 궤적을 처리하는 능력이 개선된 작업 성능으로 전환됨을 보여줍니다. 향상은 6개의 ExoRL 작업 모두에서 일관되며, Walker Run(+4.57)과 Cartpole Swingup(+2.89)에서 가장 큰 개선을 보입니다.

**장거리 검색.** 라인 검색 평가(그림 3)는 설득력 있는 정성적 결과를 제공합니다. RingAttention으로 512K 컨텍스트 길이로 미세 조정된 LLaMA-13B 모델은 500K 토큰까지 검색 작업에서 거의 완벽한 정확도를 유지하는 반면, GPT-3.5-turbo-16K는 16K에서 우연 수준으로 떨어지고, Vicuna-13B-16K는 유사하게 실패하며, Claude-2-100K는 100K를 넘어서 저하됩니다. 이는 RingAttention이 긴 컨텍스트에서 학습을 가능하게 할 뿐만 아니라 결과 모델이 실제로 해당 확장된 컨텍스트를 효과적으로 활용할 수 있음을 보여줍니다.

실험적 검증은 여러 하드웨어 구성, 모델 크기, 작업 유형을 포괄하여 철저합니다. 컨텍스트 확장 결과는 이론적 $N_h$배 향상과 정확히 일치하므로 특히 설득력이 있습니다. 그러나 다양한 컨텍스트 길이에서의 보다 포괄적인 언어 모델링 perplexity 평가와, 블록 크기가 학습 역학 및 모델 품질에 미치는 영향에 대한 절제 연구가 있으면 논문이 더 향상될 것입니다.

## 4. Critical Assessment

### Strengths
1. **우아한 이론적 기초.** 이 방법은 제로 오버헤드 통신 중첩을 달성하기 위해 자연스러운 수학적 속성(blockwise softmax의 순열 불변성)을 활용하며, 이는 엔지니어링 해킹이 아닌 깔끔하고 원칙적인 접근법입니다.
2. **장치 수에 대한 선형 확장.** 컨텍스트 길이가 장치 수에 정확히 비례하여 확장되며, 수익 감소가 없어 대규모 클러스터에서 거의 임의의 컨텍스트 크기를 가능하게 합니다.
3. **정확한 attention 계산.** attention을 근사하는 많은 긴 컨텍스트 접근법(희소 attention, 선형 attention)과 달리, RingAttention은 정확한 전체 attention을 계산하여 모델 품질 보장을 유지합니다.
4. **기존 병렬화와의 조합 가능성.** RingAttention은 FSDP, 텐서 병렬화, 파이프라인 병렬화와 직교하여, 실무자가 특정 하드웨어 및 모델 구성에 최적으로 전략을 결합할 수 있게 합니다.
5. **포괄적인 실증적 검증.** 논문은 5개의 하드웨어 구성(A100 NVLink, A100 InfiniBand, TPUv3, TPUv4, TPUv5e), 4개의 모델 크기(3B-30B), 여러 작업 유형(컨텍스트 확장, MFU, RL, 검색)에 걸쳐 평가하고 작동하는 JAX 구현을 제공합니다.
6. **실용적인 배포 가이드.** 논문은 구체적인 하드웨어 요구 사항(표 2), 병렬화 구성 권장 사항, FLOPs 확장 분석(그림 5)을 제공하여 실무자에게 실행 가능하게 만듭니다.

### Limitations
1. **최소 블록 크기 요구 사항.** 중첩 조건 $c \geq F/B$는 저대역폭 인터커넥트(예: 12.5 GB/s의 InfiniBand)에서 최소 장치당 시퀀스 길이가 매우 클 수 있음(150K)을 의미하며, 이는 상용 GPU 클러스터에서 접근법의 적용 가능성을 제한합니다.
2. **이차 FLOPs 비용은 여전히 남음.** 메모리 병목은 제거되었지만, 전체 attention의 계산 비용은 여전히 컨텍스트 길이에 대해 이차적으로 증가합니다. 극도로 긴 컨텍스트(100M+)의 경우, 그림 5에서 보여주듯이 데이터셋당 수천 배의 FLOPs가 발생할 수 있습니다.
3. **제한된 다운스트림 평가.** 언어 모델링 평가는 단일 미세 조정 모델에 대한 라인 검색 작업으로 제한됩니다. 다양한 긴 컨텍스트 벤치마크(예: 긴 문서 QA, 요약, 다중 문서 추론)에 대한 보다 포괄적인 평가가 주장을 강화할 것입니다.
4. **근사 attention 방법과의 비교 없음.** 논문은 정확한 attention 기준선과만 비교합니다. 효율적인 근사(희소 attention, 선형 attention, 상태 공간 모델)와의 다운스트림 작업 비교가 정확한 긴 컨텍스트 attention이 실제로 필요한 시점을 맥락화할 것입니다.
5. **배치 크기 트레이드오프가 완전히 탐구되지 않음.** RingAttention에 장치를 사용하면 효과적인 배치 크기가 감소합니다(동일한 총 토큰이 더 적은 시퀀스와 더 긴 시퀀스에 분산되기 때문). 이 트레이드오프가 학습 역학 및 수렴에 미치는 영향은 분석되지 않았습니다.
6. **단일 프레임워크 구현.** 구현과 실험은 JAX에만 국한됩니다. 알고리즘은 프레임워크에 구애받지 않지만, PyTorch 구현이 없으면 더 넓은 커뮤니티에서 즉각적인 채택이 제한됩니다.

### Future Directions
1. **준이차 attention과의 통합.** RingAttention을 희소 attention이나 슬라이딩 윈도우 attention과 같은 방법과 결합하면 FLOPs를 관리 가능하게 유지하면서 컨텍스트를 더욱 확장하여 실용적인 스위트 스팟을 만들 수 있습니다.
2. **적응형 블록 크기 조정.** 런타임 계산 대 대역폭 측정을 기반으로 블록 크기를 동적으로 조정하면 이질적인 하드웨어에서 중첩 조건을 최적화할 수 있습니다.
3. **다중 모달 긴 컨텍스트 응용.** 거의 무한한 컨텍스트 기능은 데이터가 본질적으로 장기간 형태인 비디오 이해, 오디오 처리, 다중 모달 추론 작업에 특히 유망합니다.
4. **계층적 링 토폴로지.** 이질적인 인터커넥트(빠른 노드 내, 느린 노드 간)를 가진 매우 큰 클러스터의 경우, 2단계 링이 두 스케일 모두에서 통신을 최적화할 수 있습니다.
5. **KV 캐시 압축과의 통합.** RingAttention을 KV 캐시 양자화 또는 제거와 같은 기술과 결합하면 블록당 통신 볼륨을 줄여 추론 시간 컨텍스트를 더욱 확장할 수 있습니다.
