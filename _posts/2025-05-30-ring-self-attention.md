---
title: "Ring Self-Attention"
date: 2025-05-30
last_modified_at: 2025-05-30
layout: post
permalink: /blog/ring-self-attention/
description: "Ring 통신 패턴을 활용한 GPU 간 시퀀스 병렬화로 분산 어텐션을 수행하는 Ring Self-Attention."
tags: sequence-parallelism ring-attention distributed-training long-sequence-modeling memory-efficient-training
thumbnail: assets/img/blog/ring-self-attention.png
series: sequence-parallelism
series_order: 1
series_title: "Sequence Parallelism Series"
related_posts: true
disqus_comments: false
giscus_comments: true
toc:
  sidebar: left
---

# TL;DR

> 본 논문은 입력 시퀀스를 여러 GPU에 분할하고 Ring Self-Attention(RSA)을 도입하여 디바이스 간 attention을 계산함으로써, 긴 시퀀스를 가진 Transformer 학습을 위한 분산 시스템 접근법인 Sequence Parallelism을 제안한다.
> 핵심 혁신은 긴 시퀀스 문제를 알고리즘적 근사가 아닌 시스템 분할 문제로 다루는 것이며, 기존의 데이터, 파이프라인, 텐서 병렬화와 호환되어 4D 병렬화를 가능하게 한다.
> 최대 64개의 NVIDIA P100 GPU에서의 실험 결과, 텐서 병렬화 대비 최대 배치 크기 $13.7\times$, 최대 시퀀스 길이 $3.0\times$ 향상을 달성했으며, sparse attention과 결합 시 114K 토큰을 초과하는 시퀀스를 처리할 수 있다.
> 주요 한계점은 attention 레이어에서의 통신 오버헤드 증가이나, MLP 레이어에서의 통신 제거로 상쇄된다.

- Paper Link: [https://arxiv.org/pdf/2105.13120](https://arxiv.org/pdf/2105.13120)

---

# Related Papers

**시퀀스 병렬화 발전:**
- [Blockwise RingAttention](/blog/blockwise_ringattention/) - 링 기반 시퀀스 병렬화의 현대적 발전
- [DeepSpeed Ulysses](https://arxiv.org/pdf/2309.14509) - 시퀀스 병렬화의 실용적 구현
- [USP](/blog/usp/) - 통합 시퀀스 병렬화 프레임워크

**분산 어텐션:**
- [DISTFLASHATTN](https://arxiv.org/pdf/2310.03294) - 분산 FlashAttention 구현
- [Striped Attention](https://arxiv.org/pdf/2311.09431) - 효율적인 시퀀스 분배 패턴
- [LoongTrain](https://arxiv.org/pdf/2406.18485) - 2D 어텐션 병렬화

**병렬화 통합:**
- [Tensor Parallelism](/blog/tp/) - 텐서 병렬화와의 결합
- [GPipe](/blog/pp/) - 파이프라인 병렬화와의 통합
- [Reducing Activation Recomputation in Large Transformer Models](/blog/sp/) - 메모리 효율적인 병렬 훈련

**긴 컨텍스트 처리:**
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/pdf/2411.01783) - 추론 시 시퀀스 병렬화

---

# Takeaways

## 1. 기여

본 연구의 근본적 동기는 시퀀스 길이에 대한 self-attention의 이차적(quadratic) 메모리 요구량이다. 표준 Transformer 구현에서는 전체 $Q$, $K$, $V$ 행렬과 attention 점수 행렬을 단일 디바이스에 저장해야 하므로, 최대 시퀀스 길이가 해당 디바이스의 메모리 용량에 본질적으로 제한된다. 이 문제에 대한 기존 접근법들은 주로 알고리즘적 근사에 집중해왔다. BigBird와 Linformer 같은 sparse attention 메커니즘은 attention의 이론적 복잡도를 $O(L^2)$에서 $O(L)$ 또는 $O(L \log L)$로 줄이지만, attention 계산 자체를 변형함으로써 모든 토큰 쌍에 대한 attention 능력을 종종 희생한다. 저자들은 이러한 알고리즘적 관점이 가치 있지만, 보완적인 시스템 수준의 해결책을 간과하고 있다고 관찰한다. 즉, attention을 근사하는 대신 정확한 attention 계산을 여러 디바이스에 분산시킬 수 있다는 것이다.

이 관찰은 긴 시퀀스가 불가피한 실제 응용 분야에서 특히 중요하다. 예를 들어 의료 영상에서는 $512 \times 512 \times 512$ 복셀 규모의 볼륨이 포함되며, 이를 토큰화하면 일반적인 NLP 입력보다 수백 배 긴 시퀀스가 생성된다. 마찬가지로 Vision Transformer(ViT)를 사용하는 고해상도 비전 태스크에서는 큰 이미지를 긴 패치 시퀀스로 변환한다. 이러한 도메인에서 시퀀스 길이 제한은 모델 병렬화(hidden 또는 depth 차원 분할)나 데이터 병렬화(배치 차원 분할)로는 직접 해결할 수 없는 실질적 병목이다.

본 논문의 핵심 기여는 입력 시퀀스를 시퀀스 차원을 따라 $N$개의 GPU에 분할하는 sequence parallelism의 설계 및 구현이다. 각 GPU는 길이 $L/N$의 부분 시퀀스만 보유하며, 모든 디바이스가 동일한 모델 파라미터를 공유한다. 이로 인해 발생하는 핵심 기술적 과제는 토큰들이 서로 다른 디바이스에 존재할 때 attention 점수와 출력을 계산하는 것이다. 이를 해결하기 위해 저자들은 Ring Self-Attention(RSA)을 제안한다. 이는 key와 value 임베딩을 링 토폴로지로 GPU 간에 순환시키는 통신 프로토콜이다. 이 접근법은 각 query가 결국 모든 key와 value에 대해 attention을 계산할 수 있도록 보장하면서, 단일 디바이스가 전체 시퀀스를 저장할 필요가 없게 한다.

핵심적인 아키텍처 이점은 호환성이다. Attention head 수(일반적으로 12-16개의 작은 수)에 의해 제약받는 텐서 병렬화와 달리, sequence parallelism의 유일한 제약 조건은 시퀀스 길이가 병렬 크기로 나누어떨어져야 한다는 것이다. 시퀀스 길이는 일반적으로 head 수보다 훨씬 크므로(예: 512 vs. 12), 이를 통해 sequence parallelism은 훨씬 더 많은 디바이스로 확장할 수 있다. 또한 이 접근법은 기존 병렬화 전략(데이터 병렬화, 파이프라인 병렬화, 텐서 병렬화)과 통합되도록 설계되어, 저자들이 4D 병렬화라고 부르는 것을 가능하게 한다. 구현은 완전히 PyTorch 기반이며, 커스텀 컴파일러나 특수 라이브러리가 필요 없어 도입 장벽을 크게 낮춘다.

실질적인 개선 효과도 상당하다. BERT Base에서 64개의 P100 GPU를 사용한 경우, sequence parallelism은 텐서 병렬화(Megatron-LM) 대비 최대 배치 크기 $13.7\times$, 최대 시퀀스 길이 $3.0\times$ 향상을 달성한다. Sparse attention(Linformer)과 결합하면 114K 토큰을 초과하는 시퀀스를 처리할 수 있으며, 이는 기존 sparse attention 구현이 단일 디바이스에서 처리 가능한 것보다 $27\times$ 긴 것이다. Weak scaling 실험에서의 일정한 메모리 스케일링 양상은 이 접근법이 근본적으로 건전함을 추가로 보여준다. 더 많은 디바이스를 추가하면 기존 메모리 오버헤드를 단순히 재분배하는 것이 아니라 비례적으로 더 긴 시퀀스를 실제로 가능하게 한다.

## 2. 방법론

### 2.1 핵심 직관

Sequence parallelism의 이론적 기반은 self-attention 계산 구조에 대한 단순하지만 강력한 관찰에 기반한다. 표준 attention 수식은 다음과 같다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q \in \mathbb{R}^{L \times d_k}$, $K \in \mathbb{R}^{L \times d_k}$, $V \in \mathbb{R}^{L \times d_v}$이다. 핵심적인 통찰은 각 query의 attention 출력 계산이 다른 query들과 독립적이라는 점이다. 구체적으로, $i$번째 query $q_i$에 대한 attention 출력은 $q_i$가 모든 key 및 value와 상호작용하는 것에 의존하지만, 이 상호작용은 key와 value의 부분 집합에 대한 부분 계산으로 분해될 수 있다. 시퀀스를 $N$개의 청크로 분할하면 디바이스 $n$은 길이 $L/N$의 부분 시퀀스에 해당하는 $Q^n$, $K^n$, $V^n$을 보유한다. $Q^n$에 대한 전체 attention을 계산하려면, 디바이스 $n$은 모든 $j \in \{1, \ldots, N\}$에 대해 $Q^n (K^j)^T$를 계산한 후 해당 value 블록 $V^j$와 부분 점수들을 결합해야 한다.

이 분해는 자연스럽게 링 통신 패턴에 매핑된다. 모든 key와 value를 모든 디바이스에 수집하는 대신(이는 메모리 절약 효과를 무효화함), 링 토폴로지는 임의의 시점에 각 디바이스가 자신의 부분 시퀀스 임베딩과 이웃으로부터 수신 중인 하나의 추가 블록만 보유하도록 보장한다. $N-1$번의 통신 라운드 후, 모든 디바이스는 모든 key 블록에 대한 부분 attention 점수를 계산하게 된다. 동일한 링 패턴이 value 임베딩에도 적용되어 최종 출력을 계산한다.

정확성에 대한 수학적 정당화는 간단하다. 디바이스 $n$에 대한 전체 attention 점수 행렬 $S^n$은 열 방향 연결(column-wise concatenation) $S^n = [S^n_1, S^n_2, \ldots, S^n_N]$으로 작성할 수 있으며, 여기서 $S^n_i = Q^n (K^i)^T / \sqrt{d_k}$이다. 전체 연결된 점수 행렬에 대해 행 방향 softmax를 적용한 후, 출력은 다음과 같다:

$$O^n = S^n V = \sum_{i=1}^{N} S^n_i V^i$$

여기서 $S^n_i \in \mathbb{R}^{L/N \times L/N}$은 디바이스 $n$의 query에서 디바이스 $i$의 key로의 attention 가중치를 나타낸다. 이 분해는 정확하며 어떠한 근사도 도입되지 않는다.

### 2.2 모델 아키텍처

Sequence parallelism 시스템은 모델 자체를 수정하지 않고 표준 Transformer 아키텍처를 감싸는 형태로 동작한다. 다른 병렬화 접근법과의 핵심적인 아키텍처 차이는 다음과 같다:

```
Pipeline Parallelism:    모델 레이어를 디바이스에 분할 (depth 차원)
Tensor Parallelism:      레이어 내 가중치 행렬을 분할 (hidden 차원)
Sequence Parallelism:    입력 시퀀스를 디바이스에 분할 (sequence 차원)
```

Sequence parallelism 설계에서 각 디바이스는 모든 모델 파라미터의 완전한 사본(동일한 가중치)을 보유하지만 입력의 부분 시퀀스만 처리한다. 단일 Transformer 레이어를 통한 데이터 흐름은 다음과 같이 진행된다:

1. **입력 분할**: 입력 시퀀스 $X \in \mathbb{R}^{B \times L \times H}$가 시퀀스 차원을 따라 $N$개의 청크로 분할되며, 디바이스 $n$은 $X^n \in \mathbb{R}^{B \times L/N \times H}$를 수신한다.

2. **선형 투영(Q/K/V)**: 각 디바이스가 공유된 가중치 행렬을 사용하여 독립적으로 로컬 query, key, value 투영을 계산한다. 디바이스 $n$은 $Q^n, K^n, V^n \in \mathbb{R}^{B \times Z \times L/N \times A}$를 생성하며, 여기서 $Z$는 attention head 수이고 $A$는 head 차원이다.

3. **Ring Self-Attention**: RSA 모듈이 분산 attention을 계산한다(2.3절에서 상세 설명).

4. **MLP 블록**: 각 디바이스가 자신의 부분 시퀀스를 독립적으로 MLP를 통해 처리한다. MLP는 토큰 단위로 동작하므로(교차 토큰 상호작용 없음) 디바이스 간 통신이 필요 없다.

5. **출력**: 부분 시퀀스 출력이 디바이스들에 분산된 상태로 유지된다.

핵심적인 관찰은 MLP 블록이 각 토큰 위치에 대해 독립적으로 동작하므로 디바이스 간 통신이 전혀 필요 없다는 점이다. 이는 MLP와 attention 레이어 모두에서 all-reduce 연산을 필요로 하는 텐서 병렬화 대비 상당한 이점이다.

### 2.3 핵심 알고리즘 및 메커니즘

**Ring Self-Attention(RSA) -- 1단계: Attention 점수 계산**

RSA는 두 단계로 진행된다. 첫 번째 단계에서는 key 임베딩이 디바이스 간에 순환되어 전체 attention 점수 행렬을 계산한다. 링 내 $N$개 디바이스의 각 디바이스 $n$에서의 알고리즘은 다음과 같다:

단계 0에서 디바이스 $n$은 로컬 부분 점수 $S^n_n = Q^n (K^n)^T / \sqrt{d_k}$를 계산한다. 이후 각 단계 $t = 1, 2, \ldots, N-1$에서:

- 디바이스 $n$은 현재 key 블록 $K^{(n-t+1) \bmod N}$을 디바이스 $(n+1) \bmod N$에 전송한다.
- 디바이스 $n$은 디바이스 $(n-1) \bmod N$으로부터 key 블록 $K^{(n-t) \bmod N}$을 수신한다.
- 디바이스 $n$은 부분 점수 $S^n_{(n-t) \bmod N} = Q^n (K^{(n-t) \bmod N})^T / \sqrt{d_k}$를 계산한다.

$N-1$ 라운드 후, 각 디바이스 $n$은 완전한 attention 점수 행 $S^n = [S^n_1, \ldots, S^n_N] \in \mathbb{R}^{L/N \times L}$을 갖게 된다. 이 시점에서 각 디바이스는 점수를 정규화하기 위해 행 방향 softmax를 적용한다.

**Ring Self-Attention(RSA) -- 2단계: Attention 출력 계산**

두 번째 단계에서는 value 임베딩이 동일한 링 방식으로 순환된다. 출력은 점진적으로 누적된다:

단계 0에서 디바이스 $n$은 $O^n = S^n_n \cdot V^n$으로 초기화한다. 각 단계 $t = 1, \ldots, N-1$에서:

- 디바이스 $n$은 현재 value 블록을 다음 디바이스에 전송하고 이전 디바이스로부터 수신한다.
- 디바이스 $n$은 다음과 같이 누적한다: $O^n \leftarrow O^n + S^n_{(n-t) \bmod N} \cdot V^{(n-t) \bmod N}$

$N-1$ 라운드 후, $O^n = \sum_{i=1}^{N} S^n_i V^i$가 되며, 이는 부분 시퀀스 $n$에 대한 정확한 attention 출력이다.

**메모리 분석 -- MLP 블록**

MLP 블록에 대한 텐서 병렬화와 sequence parallelism의 메모리 비교는 각 접근법이 우세한 명확한 영역을 드러낸다. 텐서 병렬화는 가중치 행렬을 분할하여 다음을 저장한다:

$$M_{\text{TP-MLP}} = \frac{32H^2}{N} + \frac{4BLH}{N} + BLH$$

Sequence parallelism은 전체 가중치 행렬을 유지하되 부분 시퀀스만 처리한다:

$$M_{\text{SP-MLP}} = 32H^2 + \frac{5BLH}{N}$$

$BL > 32H$일 때 sequence parallelism이 더 메모리 효율적이다. 일반적인 구성(예: BERT Base의 $B=64$, $L=512$, $H=768$)에서 $BL = 32768 > 32 \times 768 = 24576$이므로 이 이점이 확인된다.

**메모리 분석 -- Multi-Head Attention 블록**

Attention 블록의 경우, 텐서 병렬화는 attention head를 디바이스에 분할한다:

$$M_{\text{TP-Attn}} = \frac{16AZH}{N} + \frac{4BLZA}{N} + \frac{BZL^2}{N} + BLH$$

Sequence parallelism은 시퀀스 길이를 따라 분할한다:

$$M_{\text{SP-Attn}} = 16AZH + \frac{4BZLA}{N} + \frac{BZL^2}{N} + \frac{BLH}{N}$$

$BL > 16AZ$일 때 sequence parallelism이 더 메모리 효율적이다. $BZL^2/N$ 항(attention 점수 행렬)은 두 접근법에서 동일하게 스케일링되므로, 이차 성분에 대해서는 어느 쪽도 본질적인 이점이 없다.

**통신 비용 분석**

총 통신량은 sequence parallelism과 텐서 병렬화 간에 분석적으로 동일하다. Sequence parallelism에서 순전파 통신은 $2(N-1) \cdot B \cdot Z \cdot (L/N) \cdot A$(key와 value에 대한 두 번의 링 순환)이며, 역전파 통신은 $6(N-1) \cdot B \cdot Z \cdot (L/N) \cdot A$(두 번의 all-reduce와 두 번의 링 순환)이다. 합계: $8(N-1) \cdot B \cdot Z \cdot (L/N) \cdot A$.

텐서 병렬화의 경우, 4번의 집합 통신(MLP와 attention의 순전파 및 역전파) 각각이 $2(N-1) \cdot B \cdot Z \cdot (L/N) \cdot A$를 전송하여, 합계 $8(N-1) \cdot B \cdot Z \cdot (L/N) \cdot A$이다.

그러나 sequence parallelism은 파이프라인 병렬화와 결합할 때 실질적인 이점이 있다. 활성화가 이미 시퀀스 차원을 따라 분할되어 있으므로, 파이프라인 단계 간에 추가적인 split/all-gather 연산이 필요 없어 단계당 하나의 all-gather를 절약한다.

**Sparse Attention 통합**

이 시스템은 key 차원을 $L/N$에서 고정된 $K$로 줄이는 투영 단계를 추가하여 Linformer의 sparse attention과 통합된다. Sparse attention 블록의 메모리는 다음과 같다:

$$M_{\text{Sparse}} = 2AZH + \frac{2BZLA}{N} + \frac{BZLK}{N} + \frac{BLH}{N} + 2BZKA$$

결정적으로, $L$을 포함하는 모든 항이 $N$으로 나누어지므로 메모리가 $L/N$에 대해 선형적으로 스케일링된다. 이는 디바이스를 추가함으로써 이론적으로 무한한 시퀀스 길이를 가능하게 한다.

### 2.4 구현 세부사항

구현은 표준 분산 통신 원시 연산(P2P send/receive 및 all-reduce)을 사용하여 완전히 PyTorch 위에 구축되었다. 커스텀 CUDA 커널, 컴파일러, 외부 라이브러리가 필요 없다.

**하드웨어**: 실험은 NVIDIA P100 GPU(각 16GB)가 장착된 Piz Daint 슈퍼컴퓨터에서 수행되었으며, 컴퓨팅 노드당 GPU 1개, 고대역폭 인터커넥트로 연결되어 있다.

**평가 모델**: BERT Base($H=768$, $Z=12$, $A=64$) 및 BERT Large($H=1024$, $Z=16$, $A=64$).

**스케일링 제약**: 텐서 병렬화 크기는 attention head 수에 의해 제한된다(BERT Base 최대 12, BERT Large 최대 16). Sequence parallelism 크기는 시퀀스 길이가 병렬 크기로 나누어떨어지기만 하면 되므로, 64 GPU까지 확장 가능하다.

**옵티마이저**: Megatron-LM 기준선과 일관되게 Adam을 사용한다.

**통신 패턴**: 링 토폴로지는 순전파에서 key와 value 순환을 위해 point-to-point(P2P) 통신을 사용한다. 역전파에서는 그래디언트 집계를 위한 all-reduce와 추가적인 P2P 통신이 필요하다. Attention 레이어당 총 통신 라운드는 순전파에서 $2(N-1)$회, 역전파에서 $6(N-1)$회이다.

**시간 복잡도**: 디바이스당 계산량은 attention 점수 계산에 대해 $O(BL^2A/(N))$으로 유지되지만 $N$개 디바이스에 분산된다. 링 통신은 각각 $O(BZLA/N)$ 데이터를 전송하는 $O(N)$라운드를 도입한다.

## 3. 결과

실험 평가는 네 가지 차원에서 sequence parallelism의 이점을 체계적으로 보여준다: 최대 배치 크기, 최대 시퀀스 길이, 처리량, 그리고 weak scaling 양상이다.

**최대 배치 크기 스케일링**: 시퀀스 길이 512인 BERT Base에서 병렬 크기를 4에서 64 GPU로 확장할 때, sequence parallelism은 일관되게 텐서 병렬화보다 큰 최대 배치 크기를 달성한다. 64 GPU에서 sequence parallelism은 배치 크기 1600을 지원하는 반면, 텐서 병렬화는 12 GPU 한계에서 최대 117에 그쳐 $13.7\times$ 향상을 보인다. 동일한 병렬 크기(예: 4 GPU)에서도 sequence parallelism은 약 $2\times$ 더 큰 배치 크기를 지원한다. BERT Large에서도 이 이점은 지속되며, 64 GPU에서 sequence parallelism은 16 GPU 텐서 병렬화 대비 $10.2\times$의 최대 배치 크기를 달성한다.

| 구성 | 최대 배치 크기 (TP) | 최대 배치 크기 (SP) | 비율 |
|---|---|---|---|
| BERT Base, 4 GPU | ~100 | ~200 | ~2x |
| BERT Base, 12 GPU (TP 최대) | 117 | ~450 | ~3.8x |
| BERT Base, 64 GPU | 해당 없음 | 1600 | 13.7x vs TP@12 |
| BERT Large, 16 GPU (TP 최대) | ~47 | ~127 | ~2.7x |
| BERT Large, 64 GPU | 해당 없음 | ~480 | 10.2x vs TP@16 |

**처리량**: 동일한 병렬 크기에서 sequence parallelism은 텐서 병렬화와 유사한 처리량을 달성하며(대부분의 구성에서 5-10% 이내), 링 통신이 유의미한 오버헤드를 도입하지 않음을 보여준다. 파이프라인 병렬화와 결합하면, 텐서 병렬화 파이프라인 단계 간에 필요한 split/all-gather 연산이 제거되어 오히려 더 높은 처리량을 달성한다.

**최대 시퀀스 길이**: BERT Base에서 배치 크기를 64로 고정한 경우, 64 GPU에서 sequence parallelism은 약 2250 토큰의 시퀀스를 지원하는 반면, 텐서 병렬화는 스케일링 한계에서 약 750 토큰이 최대로 $3.0\times$ 향상을 보인다. 배치 크기 16인 BERT Large에서는 약 $2\times$ 향상이 달성된다.

**Sparse Attention 통합**: 32개의 P100 GPU에서 배치 크기 4로 Linformer sparse attention과 결합한 경우, sequence parallelism은 거의 이상적인 선형 스케일링을 달성하여 114K 토큰 이상의 시퀀스를 처리한다. 이는 기존 sparse attention 연구들이 단일 디바이스에서 처리할 수 있는 것(약 4K 토큰)보다 $27\times$ 긴 것이다. 거의 이상적인 스케일링은 $L$에 의존하는 모든 메모리 항이 실제로 $L/N$으로 스케일링됨을 확인해준다.

**Weak Scaling**: Weak scaling 실험은 이 접근법의 건전성에 대한 가장 설득력 있는 증거를 제공한다. 배치 크기를 병렬 크기에 비례하여 확장할 때(1에서 8 GPU), 텐서 병렬화의 메모리는 8477MB에서 8 GPU에서 OOM으로 증가하는 반면, sequence parallelism의 메모리는 약 8480MB로 거의 일정하게 유지된다. 이 일정한 메모리 양상은 시퀀스 의존적 메모리 항이 $1/N$으로 스케일링된다는 이론적 분석을 직접 검증한다.

| 병렬 크기 | 배치 크기 | 시퀀스 길이 | TP 메모리 (MB) | SP 메모리 (MB) | TP 토큰/초 | SP 토큰/초 |
|---|---|---|---|---|---|---|
| 1 | 64 | 512 | 8477 | 8478 | 9946 | 9261 |
| 2 | 128 | 512 | 9520 | 8479 | 15510 | 13938 |
| 4 | 256 | 512 | 12233 | 8481 | 20702 | 21270 |
| 8 | 512 | 512 | OOM | 8491 | OOM | 26402 |

**수렴 검증**: Wikipedia에서 병렬 크기 4로 BERT Large를 50K 이터레이션 학습한 결과, sequence parallelism은 masked language modeling(MLM) 손실과 sentence order prediction(SOP) 손실 모두에서 Megatron의 텐서 병렬화와 동등한 수렴을 달성하며, 약간 더 낮은 최종 손실 값을 보인다. 이는 분산 attention 계산이 학습에 영향을 미치는 수치적 부정확성을 도입하지 않음을 확인해준다.

실험 결과는 종합적으로 sequence parallelism이 시퀀스가 모델 차원 대비 긴 영역(정확히 기존 병렬화 접근법이 한계를 보이는 영역)에서 의미 있는 개선을 제공하는 실용적 시스템임을 보여준다.

## 4. 비판적 평가

### 강점
1. **우아한 시스템 수준 해결책**: 본 논문은 긴 시퀀스 문제를 알고리즘적 근사뿐만 아니라 시스템 수준에서 해결할 수 있음을 식별하여, 기존 sparse attention 방법과 결합 가능한 보완적 최적화 차원을 열었다.
2. **정확한 계산**: Sparse attention 방법과 달리, sequence parallelism은 정확한 전체 attention을 계산하여 메모리 한계를 극복하면서도 모델 품질을 보존한다. 이는 근사 기반 접근법에 내재된 정확도 트레이드오프를 피하는 원칙적인 이점이다.
3. **우수한 결합성**: 이 접근법은 데이터 병렬화, 파이프라인 병렬화, 텐서 병렬화와 호환되어 4D 병렬화를 가능하게 한다. 이러한 결합성은 모델 크기, 배치 크기, 시퀀스 길이 등 여러 자원 제약을 동시에 해결해야 하는 실제 배포 환경에서 매우 중요하다.
4. **일정한 weak scaling 메모리**: 배치 크기와 디바이스를 함께 확장할 때 메모리가 거의 일정하게 유지되는 실험적 시연은 이론적 분석의 강력한 검증이며, 더 큰 규모에서도 이 접근법이 효과적일 것임을 시사한다.
5. **실용적 구현**: 커스텀 컴파일러나 라이브러리 없이 PyTorch만으로 구현함으로써, 특수 인프라를 요구하는 접근법(예: TensorFlow의 정적 계산 그래프를 필요로 하는 GShard/GSPMD) 대비 도입 장벽을 크게 낮춘다.

### 한계점
1. **통신 오버헤드 스케일링**: 총 통신량은 텐서 병렬화와 동일하지만, 링 통신 패턴이 $O(N)$회의 순차적 라운드를 필요로 하므로 매우 많은 GPU나 높은 지연 시간의 인터커넥트에서 병목이 될 수 있다. P2P 통신은 all-reduce 연산만큼 효과적으로 오버랩될 수 없다.
2. **Softmax 장벽**: 분산된 부분 attention 점수에 대한 softmax 정규화 처리 방법이 다루어지지 않는다. Softmax 계산은 수치적 안정성을 위해 모든 key에 걸친 최대 점수를 알아야 하며, 이는 명시적으로 논의되지 않은 추가 통신을 필요로 한다.
3. **제한된 모델 평가**: 실험이 현대 기준으로 비교적 작은 모델인 BERT Base와 BERT Large에서만 수행되었다. 디코더 전용 모델(GPT 스타일)이나 훨씬 큰 모델에서의 동작은 시연되지 않았다.
4. **구형 하드웨어**: 모든 실험이 NVIDIA P100 GPU(2016년 출시, 16GB)에서 수행되어, 절대적인 수치의 현대 하드웨어 관련성이 제한된다. A100이나 H100 GPU에서는 메모리 대 연산 비율과 인터커넥트 대역폭이 다르므로 상대적 개선폭이 달라질 수 있다.
5. **인과적 attention 미처리**: 본 논문은 양방향 self-attention만 명시적으로 고려한다. 인과적(자기회귀) attention은 링 통신 패턴에서 부하 불균형을 야기할 수 있는 비대칭성을 도입하며, 이는 다루어지지 않았다.
6. **4D 병렬화 미시연**: 기여로 주장되었으나, 네 가지 병렬화 차원 모두의 실제 통합은 향후 과제로 남겨졌다. 2D 조합(시퀀스 + 파이프라인, 시퀀스 + 텐서)만 실험적으로 검증되었다.
7. **엔드투엔드 태스크 평가 부재**: 시스템 수준 지표(메모리, 처리량, 최대 배치/시퀀스 크기)만 평가되었다. 확장된 시퀀스 길이에 대한 다운스트림 태스크 성능이 보고되지 않아, 더 긴 시퀀스 처리 능력이 실질적인 응용 품질 향상으로 이어지는지 불분명하다.

### 향후 방향
1. **FlashAttention과의 통합**: Sequence parallelism을 IO-aware attention 계산(FlashAttention)과 결합하면 디바이스 간 및 디바이스 내 메모리 효율성을 동시에 개선하는 복합적 이점을 얻을 수 있다.
2. **인과적 attention 지원**: 부하 균형을 고려한 인과적(자기회귀) attention으로 RSA를 확장하면, 현대 LLM 아키텍처를 지배하는 디코더 전용 언어 모델에 대한 적용 범위가 넓어질 것이다.
3. **통신-계산 오버랩**: 다음 라운드의 key/value 전송이 현재 라운드의 계산과 오버랩되는 파이프라인 통신을 구현하면, 특히 큰 링 크기에서 통신 지연을 숨기고 처리량을 향상시킬 수 있다.
4. **이종 병렬화 스케줄링**: 모델 아키텍처와 하드웨어 토폴로지에 기반하여 데이터, 파이프라인, 텐서, 시퀀스 병렬화 차원에 걸친 디바이스 할당을 공동으로 최적화하는 자동 스케줄링 전략의 개발이 필요하다.
5. **비전 및 멀티모달 모델에의 적용**: 고해상도 의료 영상 처리나 시퀀스 길이가 자연스럽게 단일 디바이스 메모리 용량을 초과하는 기타 도메인의 ViT 기반 모델에 sequence parallelism을 적용하여, NLP를 넘어선 실질적 영향을 검증할 필요가 있다.

---
