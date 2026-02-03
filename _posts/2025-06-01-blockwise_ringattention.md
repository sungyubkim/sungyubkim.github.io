---
title: "Blockwise RingAttention"
date: 2025-06-01
last_modified_at: 2025-06-01
layout: post
permalink: /blog/blockwise_ringattention/
description: "블록 단위 계산과 링 통신을 결합하여 초장문 시퀀스 처리의 메모리 병목을 해결하는 Blockwise RingAttention."
tags: sequence-parallelism ring-attention memory-efficiency distributed-training
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

> **Blockwise RingAttention**은 AI 모델이 매우 긴 시퀀스(전체 책, 긴 동영상, 대규모 데이터셋 등)를 처리하지 못하게 하는 근본적인 메모리 병목 현상을 해결합니다. 기존 트랜스포머는 제곱 메모리 요구사항을 가집니다 - 100만 토큰을 처리하려면 1M×1M 어텐션 행렬을 저장해야 하는데, 이는 계산상 불가능합니다.
> 
> **핵심 혁신**: Blockwise RingAttention은 이 계산을 여러 장치에 "링(ring)" 토폴로지로 분산시켜, 각 장치가 시퀀스의 일부를 처리하고 다음 장치로 정보를 전달합니다. 이를 통해 **장치_수 × 기존_한계**만큼 긴 시퀀스 처리가 가능합니다.
> 
> **주요 기여점**:
> 
> 1. **거의 무한한 컨텍스트**: 1억+ 토큰 시퀀스 훈련 가능 (약 300권의 책에 해당)
> 2. **선형 확장**: 컨텍스트 길이가 장치 수에 선형적으로 확장, 시퀀스 길이에 제곱적으로 확장되지 않음
> 3. **근사 없음**: 정확도를 희생하지 않고 정확한 어텐션 계산
> 4. **메모리 효율성**: 각 장치는 전체 시퀀스의 일부만 저장
> 5. **통신 최적화**: 계산과 데이터 전송을 중첩시켜 최소한의 오버헤드
> 
> **영향**: 전체 코드베이스 처리, 완전한 영화 분석, 대규모 과학 데이터셋 훈련 등 이전에는 불가능했던 작업들을 현실화합니다.

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

## 1. 근본적 문제: 트랜스포머가 긴 시퀀스를 처리할 수 없는 이유

### 메모리 벽(Memory Wall) 문제

기존 트랜스포머 모델은 제가 "메모리 벽"이라고 부르는 제곱 확장 문제로 인해 긴 시퀀스 처리가 계산상 불가능합니다.

**문제의 수학적 분석:**

- 시퀀스 길이: 100만 토큰
- 어텐션 행렬 크기: 100만 × 100만 = 1조 개 원소
- 필요 메모리: 1조 × 4바이트 = 4TB (어텐션 행렬만으로!)

**실제 컨텍스트 비교:**

- GPT-3.5: 16K 토큰 (~32페이지 텍스트)
- Claude 2: 200K 토큰 (~400페이지)
- Gemini 1.5: 100만 토큰 (~2,000페이지)
- **Blockwise RingAttention**: 1억+ 토큰 (~20만 페이지 또는 300권 이상의 책)

### 논문의 동기 논리

저자들은 기존 접근법의 세 가지 핵심 한계를 식별합니다:

1. **메모리 병목**: 메모리 효율적 어텐션(FlashAttention)을 사용해도 레이어 출력 저장이 금지적임
2. **장치 제약**: 최적화와 관계없이 단일 장치 메모리가 컨텍스트 길이를 제한
3. **통신 오버헤드**: 기존 분산 접근법은 상당한 계산 비용 추가

**제 분석**: 논문은 해결책이 단순히 어텐션 계산 최적화가 아니라, 여러 장치에서 계산을 분산하고 조율하는 방식을 근본적으로 재고하는 것임을 훌륭하게 인식했습니다.

## 2. Blockwise RingAttention 솔루션: 혁명적 접근법

### 핵심 혁신: 링 토폴로지 + 블록별 계산

논문은 두 가지 접근법을 도입합니다:

1. **공간적 분산**: 시퀀스를 장치들에 링 형태로 분산
2. **시간적 중첩**: 통신과 계산을 중첩

**개념적 Python 코드:**

```python
def ring_attention_concept():
    # 기존 방식 (불가능):
    # 장치 1: 전체 100만 토큰 시퀀스 처리 (불가능)
    
    # Blockwise RingAttention 방식:
    장치수 = 8
    장치당_토큰수 = 1_000_000 // 장치수  # 각각 125K 토큰
    
    # 각 장치는 자신의 청크를 처리하면서 다른 장치와 K,V를 공유
    for 장치_id in range(장치수):
        로컬_토큰 = 토큰[장치_id * 장치당_토큰수:(장치_id + 1) * 장치당_토큰수]
        
        # 핵심 통찰: 각 장치는 링 통신을 통해 모든 토큰에 어텐션할 수 있지만
        # 로컬에는 시퀀스의 1/8만 저장
        process_local_chunk_with_ring_attention(로컬_토큰, 장치_id)
```

### 성공을 위한 핵심 가정과 조건

**하드웨어 요구사항 (필수):**

1. **고대역폭 인터커넥트**: NVLink (600 GB/s) 또는 InfiniBand (200 Gb/s)
2. **동기화된 장치**: 모든 장치가 동기화되어 작동해야 함
3. **충분한 장치별 메모리**: 각 장치는 로컬 시퀀스 청크 + 버퍼용 메모리 필요

**소프트웨어 요구사항:**

1. **균등한 시퀀스 분할**: 시퀀스 길이가 장치 수로 나누어떨어져야 함
2. **적절한 인과 마스킹**: 자기회귀 모델을 위한 글로벌 위치 추적
3. **수치적 안정성**: 온라인 소프트맥스 계산은 세심한 구현 필요

**제 비판적 평가**: 논문의 성공은 이러한 가정에 크게 의존합니다. 하드웨어 요구사항이 본질적으로 이 접근법을 자금이 풍부한 연구소와 대기업으로 제한한다는 점은 논문에서 충분히 다루지 않은 중요한 실용적 한계입니다.

## 3. 상세 기술 구현

### 단계별 알고리즘과 Python 예제

**1단계: 시퀀스 분할**

```python
def partition_sequence_for_ring(sequence, num_devices):
    """
    입력 시퀀스를 링 토폴로지의 장치들에 분할
    
    실제 예제: 100만 토큰 시퀀스를 8개 장치에 분할
    """
    seq_len = len(sequence)  # 1,000,000 토큰
    chunk_size = seq_len // num_devices  # 장치당 125,000 토큰
    
    partitions = {}
    for device_id in range(num_devices):
        start_idx = device_id * chunk_size
        end_idx = start_idx + chunk_size
        
        partitions[device_id] = {
            'tokens': sequence[start_idx:end_idx],
            'global_start': start_idx,  # 인과 마스킹에 중요
            'global_end': end_idx,
            'device_id': device_id
        }
    
    return partitions
```

**2단계: 링 통신 설정**

```python
def setup_ring_communication(num_devices):
    """
    링 토폴로지 생성: 0 -> 1 -> 2 -> ... -> N-1 -> 0
    
    원형으로 메모를 전달하는 것처럼 생각하면 됨
    """
    ring = {}
    for device_id in range(num_devices):
        ring[device_id] = {
            'next_device': (device_id + 1) % num_devices,
            'prev_device': (device_id - 1) % num_devices,
            'position_in_ring': device_id
        }
    
    return ring
```

**3단계: 핵심 Blockwise RingAttention 계산**

```python
def ring_attention_core(local_qkv, ring_topology, device_id):
    """
    Blockwise RingAttention의 핵심: 장치 링에서 어텐션 계산
    
    핵심 혁신: 전체 어텐션 행렬을 구체화하지 않고 온라인 소프트맥스 계산
    """
    Q_local = local_qkv['Q']  # 형태: [local_seq_len, d_model]
    K_local = local_qkv['K'] 
    V_local = local_qkv['V']
    
    # 온라인 소프트맥스용 누적기 초기화
    output = torch.zeros_like(Q_local)
    row_max = torch.full((Q_local.shape[0],), float('-inf'))
    row_sum = torch.zeros(Q_local.shape[0])
    
    num_devices = ring_topology[device_id]['total_devices']
    
    # 링의 각 장치의 K,V 처리
    for ring_step in range(num_devices):
        kv_device_id = (device_id + ring_step) % num_devices
        
        if ring_step == 0:
            # 첫 번째 단계: 로컬 K,V 사용
            K_current = K_local
            V_current = V_local
        else:
            # 후속 단계: 링에서 K,V 수신
            K_current, V_current = receive_from_ring(ring_topology, device_id)
        
        # 이 K,V 블록에 대한 어텐션 점수 계산
        scores = Q_local @ K_current.transpose(-2, -1)
        scores = scores / math.sqrt(Q_local.shape[-1])  # 안정성을 위한 스케일링
        
        # 글로벌 위치 기반 인과 마스킹 적용
        scores = apply_global_causal_mask(scores, local_qkv, kv_device_id)
        
        # 온라인 소프트맥스 업데이트 (핵심 혁신)
        output, row_max, row_sum = online_softmax_update(
            output, row_max, row_sum, scores, V_current
        )
        
        # 중첩: 계산하면서 현재 K,V를 다음 장치로 전송 시작
        if ring_step < num_devices - 1:
            async_send_to_ring(K_current, V_current, ring_topology, device_id)
    
    return output
```

**제 기술적 통찰**: 이 접근법의 훌륭함은 온라인 소프트맥스 계산에 있습니다. 기존 어텐션은 전체 어텐션 행렬 저장이 필요하지만, Blockwise RingAttention은 점진적 업데이트가 가능한 실행 통계를 유지합니다. 이는 수학적으로 동등하지만 훨씬 메모리 효율적입니다.

## 4. 실험 결과 및 분석

### 주요 성능 결과

|하드웨어 설정|모델 크기|기준선 (BPT)|Ring Attention|**개선도**|**제 해석**|
|---|---|---|---|---|---|
|8×A100 GPU|7B|128K 토큰|**100만+ 토큰**|**8배 개선**|_단일 노드 훈련 역량 변혁_|
|32×A100 GPU|7B|512K 토큰|**1600만+ 토큰**|**32배 개선**|_전체 코드베이스나 책 처리 가능_|
|TPUv4-512|7B|256K 토큰|**6500만+ 토큰**|**256배 개선**|_혁명적 규모 - 130권 이상 소설에 해당_|
|TPUv4-1024|7B|256K 토큰|**1억3천만+ 토큰**|**512배 개선**|_실용적 목적의 거의 무한한 컨텍스트_|

**제 분석**: 이러한 결과는 패러다임 전환을 나타냅니다. TPUv4-1024에서의 1억3천만 토큰 역량은 전체 해리포터 시리즈(110만 단어 ≈ 150만 토큰)를 87번 반복해서 단일 컨텍스트 윈도우에서 처리할 수 있음을 의미합니다. 이는 완전히 새로운 연구 방향을 열어줍니다.

### 모델 FLOPs 활용률(MFU) 분석

|하드웨어|컨텍스트 길이|BPT MFU|Ring Attention MFU|**효율성 증가**|**제 평가**|
|---|---|---|---|---|---|
|8×A100|512K|28×10³|**240×10³**|**8.6배 향상**|_뛰어난 컴퓨팅 효율성_|
|32×A100|1M|24×10³|**224×10³**|**9.3배 향상**|_규모에서 효율성 유지_|
|TPUv4-512|16M|1024×10³|**8192×10³**|**8.0배 향상**|_TPU에서 일관된 확장_|

**핵심 통찰**: 8-9배 MFU 개선은 Blockwise RingAttention이 단순히 더 긴 컨텍스트를 가능하게 할 뿐만 아니라 이전 방법보다 더 효율적으로 수행한다는 것을 나타내므로 놀랍습니다. 이는 통신 오버헤드가 영리한 중첩을 통해 성공적으로 숨겨졌음을 시사합니다.

### 제거 연구(Ablation Study) 결과

|제거된 구성요소|최대 컨텍스트|성능 영향|**제 해석**|
|---|---|---|---|
|**링 통신 없음**|128K|기준선|_링 토폴로지가 필수임을 확인_|
|**통신 중첩 없음**|1M|45% 느림|_중첩 전략이 중요함을 증명_|
|**최적이 아닌 블록 크기**|1M|20% 느림|_블록 크기 튜닝이 상당히 중요_|
|**전체 시스템**|**1M+**|**최적**|_모든 구성요소가 시너지 효과를 발휘_|

**제 비판적 평가**: 제거 연구는 각 구성요소가 필요하다는 것을 설득력 있게 보여줍니다. 하지만 실패 모드와 엣지 케이스에 대한 더 많은 분석을 보고 싶었습니다.

### 강화학습 검증

|방법|AntMaze-Large|Kitchen-Mixed|Adroit-Human|**평균**|**제 평가**|
|---|---|---|---|---|---|
|BC 기준선|0.45|0.32|0.28|**0.35**|_표준 성능_|
|Decision Transformer|0.52|0.41|0.35|**0.43**|_적당한 개선_|
|AT + BPT|0.65|0.58|0.51|**0.58**|_강력한 기준선_|
|**AT + Ring Attention**|**0.72**|**0.64**|**0.58**|**0.65**|_**12% 개선 - 상당함**_|

**제 분석**: RL 결과는 Blockwise RingAttention의 이점이 언어 모델링을 넘어 확장된다는 것을 보여주므로 특히 설득력이 있습니다. 12% 개선은 더 긴 컨텍스트가 진정으로 순차적 의사결정을 개선한다는 것을 시사하며, 이는 로봇공학과 자율 시스템에 깊은 의미를 가집니다.

## 5. 핵심 조건과 한계

### 하드웨어 전제조건 (논문의 아킬레스건)

**대역폭 요구사항 분석:**

- 시퀀스 길이: 100만 토큰
- 모델 차원: 4096
- 장치 수: 8개
- 장치당 필요 대역폭: ~150 GB/s

**실제 하드웨어와의 비교:**

- NVLink (600 GB/s): ✓ 호환
- InfiniBand (25 GB/s): ✗ 부족
- 이더넷 (1.25 GB/s): ✗ 심각하게 부족
- PCIe (8 GB/s): ✗ 심각하게 부족

**제 비판적 평가**: 논문은 하드웨어 요구사항을 과소평가합니다. NVLink나 고급 InfiniBand가 본질적으로 필수이므로 실용적 채택이 자금이 풍부한 기관으로 제한됩니다.

### 동기화 문제

**숨겨진 복잡성:**

1. 모든 장치가 각 링 단계를 완료한 후에만 다음 단계 시작 가능
2. 장치 실패나 속도 저하가 전체 파이프라인 차단
3. 이질적 하드웨어에서 부하 분산이 중요해짐
4. 네트워크 지터가 연쇄 지연 유발 가능

**제 통찰**: 논문은 "가장 약한 고리" 문제를 충분히 다루지 않습니다. 실제로는 하나의 느린 장치가 전체 성능을 크게 감소시킬 수 있습니다.

## 6. 광범위한 의미와 미래 방향

### 가능해진 혁신적 응용

**과학 컴퓨팅:**

- **유전체학**: 전체 게놈(30억 염기쌍)을 단일 컨텍스트에서 처리
- **기후 모델링**: 수십 년의 연속 센서 데이터 분석
- **천문학**: 수년간의 망원경 관측을 동시에 처리

**창조 산업:**

- **영화 분석**: 대화와 함께 전체 영화를 프레임별로 처리
- **문학**: 완전한 책 시리즈의 주제와 패턴 분석
- **음악**: 전체 오케스트라 컨텍스트로 교향곡 작곡

**소프트웨어 엔지니어링:**

- **코드베이스 분석**: 수백만 줄의 코드를 동시에 처리
- **버그 탐지**: 전체 프로젝트 히스토리에서 패턴 분석
- **문서화**: 전체 코드베이스에서 포괄적 문서 생성

### 미래 연구에 대한 제 예측

1. **하이브리드 접근법**: 더 큰 효율성을 위해 Blockwise RingAttention과 스파스 어텐션 패턴 결합
2. **동적 링 토폴로지**: 워크로드 특성에 반응하는 적응형 링 구조
3. **계층적 링 시스템**: 극한 규모 배포를 위한 다단계 링
4. **하드웨어 공동 설계**: 링 통신 패턴에 최적화된 맞춤형 인터커넥트

### 논문이 다뤄야 할 한계

**경제적 실현가능성**: 하드웨어 비용으로 인해 이 접근법이 대기업에만 접근 가능해져 AI 불평등을 악화시킬 수 있음

**에너지 소비**: 1억+ 토큰 시퀀스 처리의 환경 영향을 분석하지 않음

**내결함성**: 장치 실패나 네트워크 분할을 시스템이 어떻게 처리하는지에 대한 논의 없음

## 7. 최종 평가

### 논문이 옳게 한 것

1. **혁명적 규모**: 1억+ 토큰 컨텍스트 달성은 진정으로 혁신적
2. **엄격한 엔지니어링**: 온라인 소프트맥스와 통신 중첩은 훌륭한 최적화
3. **광범위한 검증**: 언어 모델링과 RL에서의 결과가 일반화 가능성 입증
4. **선형 확장**: 장치 수 확장이 우아하고 실용적

### 개선될 수 있는 것

1. **접근성**: 엘리트 기관을 넘어 이 기술을 어떻게 사용 가능하게 할지에 대한 더 많은 논의
2. **실패 모드**: 엣지 케이스와 시스템 견고성 분석
3. **에너지 효율성**: 환경 영향 고려사항
4. **구현 세부사항**: 재현을 시도하는 실무자를 위한 더 많은 지침

### 제 전체 평가

Blockwise RingAttention은 완전히 새로운 범주의 AI 응용을 가능하게 할 시퀀스 모델링의 근본적 돌파구를 나타냅니다. 하드웨어 요구사항이 즉시 채택을 제한하지만, 분산 어텐션 계산에 대한 핵심 통찰은 차세대 AI 시스템에 영향을 미칠 가능성이 높습니다.

이 논문은 트랜스포머의 메모리 벽이 영리한 분산 컴퓨팅을 통해 극복될 수 있음을 성공적으로 보여주어, 이전에는 불가능했던 규모에서 정보를 처리하고 추론할 수 있는 AI 시스템으로의 문을 열었습니다. 이 작업은 더 능력 있고 포괄적인 AI 시스템으로의 진화에서 중요한 순간으로 기억될 가능성이 높습니다.

**중요도 점수: 9/10** - AI에서 가능한 것을 근본적으로 바꾸는 드문 논문으로, 명확한 실용적 영향과 광범위한 적용 가능성을 가짐.

---