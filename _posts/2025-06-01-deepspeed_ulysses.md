---
title: "DeepSpeed Ulysses"
date: 2025-06-01
last_modified_at: 2025-06-01
layout: post
permalink: /blog/deepspeed_ulysses/
description: "DeepSpeed-Ulysses achieves efficient sequence parallelism via all-to-all communication for 1M+ token training."
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

> DeepSpeed-Ulysses는 대형 Transformer 모델 훈련에서 "시퀀스 차원 스케일링 문제"를 해결하는 새로운 시퀀스 병렬처리 접근법을 제시합니다. **주요 기여:** (1) 영리한 데이터 재분배를 통해 4배 더 긴 시퀀스(1M+ 토큰) 훈련 가능, (2) 기존 O(N) 방법 대비 O(N/P) 통신 복잡도 달성, (3) 모델 품질 유지하면서 구성에 따라 1.5-3.5배 속도 향상 제공. **핵심 혁신:** all-to-all 통신을 사용하여 시퀀스 병렬 데이터를 헤드 병렬 계산으로 변환, 각 GPU가 어텐션 헤드의 부분집합에 대해 완전한 어텐션을 계산할 수 있게 함. **중요한 제약:** 구성에 따라 성능 향상이 크게 다름, sparse attention에서 성능 저하, 실험 검증의 통계적 엄밀성 부족.

- Paper Link: [https://arxiv.org/pdf/2309.14509](https://arxiv.org/pdf/2309.14509)

---

# Related Papers

**시퀀스 병렬화 방법론:**
- [Blockwise RingAttention](/blog/blockwise_ringattention/) - 링 토폴로지를 활용한 시퀀스 병렬화
- [Ring Self-Attention](/blog/ring-self-attention/) - 시퀀스 병렬화 종합 분석
- [USP](/blog/usp/) - Ulysses와 Ring을 통합한 시퀀스 병렬화

**긴 시퀀스 훈련:**
- [LoongTrain](https://arxiv.org/pdf/2406.18485) - 2D 어텐션을 활용한 긴 시퀀스 훈련
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/pdf/2411.01783) - 컨텍스트 병렬화를 통한 추론

**어텐션 최적화:**
- [DISTFLASHATTN](https://arxiv.org/pdf/2310.03294) - 분산 FlashAttention 구현
- [Striped Attention](https://arxiv.org/pdf/2311.09431) - 효율적인 어텐션 분배 패턴

**시스템 통합:**
- [Tensor Parallelism](/blog/tp/) - 텐서 병렬화와의 결합
- [Reducing Activation Recomputation in Large Transformer Models](/blog/sp/) - 메모리 효율적인 훈련

---

# Takeaways

문제 정의: 시퀀스 스케일링이 중요한 이유

현재 AI 애플리케이션들은 점점 더 긴 컨텍스트 추론을 요구합니다:
- **대화형 AI**: 확장된 대화에서 컨텍스트 유지
- **문서 분석**: 전체 책 처리 (100K+ 단어)
- **과학 컴퓨팅**: 유전체 시퀀스 분석 (수십억 염기쌍)
- **비디오 생성**: 긴 시퀀스에서 시간적 관계 이해

하지만 기존 병렬처리 전략들은 시퀀스 차원에서 실패합니다:
- **데이터 병렬처리**: 배치 크기 확장, 시퀀스 길이 아님
- **텐서 병렬처리**: 모델 너비 확장, 시퀀스 길이 아님
- **파이프라인 병렬처리**: 모델 깊이 확장, 시퀀스 길이 아님

**근본적 도전**: 어텐션 계산은 O(N²) 메모리 복잡도를 가져 긴 시퀀스를 처리하기에 비용이 너무 큽니다.

## 해결책: 어텐션 중심 시퀀스 병렬처리

### 핵심 혁신: 데이터 재분배 전략

DeepSpeed-Ulysses는 영리한 통찰을 통해 시퀀스 병렬처리 문제를 변환합니다: 어텐션 연산 내에서 병렬화를 시도하는 대신, 데이터를 재분배하여 헤드 간 병렬 어텐션 계산을 가능하게 합니다.

```python
def deepspeed_ulysses_pipeline(input_tokens, num_heads, world_size):
    """
    핵심 변환을 보여주는 완전한 파이프라인
    """
    # 단계 1: 디바이스 간 시퀀스 분할
    # 입력: [batch, full_seq_len, hidden_dim]
    # 결과: 각 디바이스가 [batch, seq_len/P, hidden_dim] 보유
    local_input = partition_sequence(input_tokens, world_size)
    
    # 단계 2: QKV 로컬 계산 (embarrassingly parallel)
    local_q, local_k, local_v = compute_qkv(local_input)
    
    # 단계 3: ALL-TO-ALL 변환 (핵심 혁신)
    # 변환: 시퀀스 병렬 → 헤드 병렬
    # 이전: 각 디바이스가 부분 시퀀스, 모든 헤드 보유
    # 이후: 각 디바이스가 전체 시퀀스, 헤드 부분집합 보유
    global_q, global_k, global_v = all_to_all_redistribute(
        local_q, local_k, local_v, num_heads, world_size
    )
    
    # 단계 4: 할당된 헤드에서 어텐션 계산 (디바이스 간 병렬)
    # 각 디바이스가 num_heads/P 헤드에 대해 완전한 어텐션 계산
    attention_output = compute_attention_heads(global_q, global_k, global_v)
    
    # 단계 5: ALL-TO-ALL 복원 (시퀀스 병렬처리 복원)
    # 변환: 헤드 병렬 → 시퀀스 병렬
    final_output = all_to_all_restore(attention_output, world_size)
    
    return final_output
```

### 작동 원리: 수학적 통찰

핵심 통찰은 multi-head attention이 자연스럽게 분해 가능하다는 것입니다:
```python
# 표준 어텐션: 모든 헤드를 함께 계산
def standard_attention(Q, K, V, num_heads):
    # Q, K, V: [batch, seq_len, hidden_dim]
    outputs = []
    for head in range(num_heads):
        q_h = Q[:, :, head*head_dim:(head+1)*head_dim]
        k_h = K[:, :, head*head_dim:(head+1)*head_dim]  
        v_h = V[:, :, head*head_dim:(head+1)*head_dim]
        
        # 이 계산은 헤드 간에 독립적입니다!
        attn_h = softmax(q_h @ k_h.T / sqrt(head_dim)) @ v_h
        outputs.append(attn_h)
    
    return concat(outputs)

# DeepSpeed-Ulysses: 디바이스 간 헤드 분산
def distributed_attention(Q, K, V, device_heads):
    # 각 디바이스는 할당된 헤드만 계산
    # 하지만 해당 헤드들에 대해 전체 시퀀스를 봄
    outputs = []
    for head in device_heads:  # 전체 헤드의 부분집합
        attn_h = compute_single_head_attention(Q, K, V, head)
        outputs.append(attn_h)
    
    return concat(outputs)
```

## 중요한 가정과 조건

### 수학적 요구사항
```python
# 방법이 작동하기 위한 하드 제약:
assert sequence_length % world_size == 0  # 균등 분할 필수
assert num_heads % world_size == 0        # 헤드 균등 분산 필수
assert hidden_dim % num_heads == 0        # 표준 어텐션 요구사항

# 실패 사례 예시:
sequence_length = 1000  # world_size=8로 나누어떨어지지 않음
# 패딩이나 불균등 분산이 필요하여 구현 복잡화
```

### 인프라 요구사항
```python
def validate_infrastructure(bandwidth_gbps, latency_ms, world_size):
    """
    All-to-all 통신 효율성은 네트워크 토폴로지에 크게 의존
    """
    # 경험법칙: 높은 bisection bandwidth 필요
    required_bandwidth = world_size * 10  # GB/s per device
    if bandwidth_gbps < required_bandwidth:
        return False, "네트워크 대역폭 부족"
    
    if latency_ms > 1.0:  # 높은 지연시간은 작은 메시지 성능 저하
        return False, "네트워크 지연시간 과다"
    
    return True, "인프라 적합"
```

### 부하 균형 가정
이 방법은 다음에 대한 균등한 계산 부하를 가정합니다:
- 시퀀스 청크 (구조화된 데이터에서는 성립하지 않을 수 있음)
- 어텐션 헤드 (일반적으로 참이지만 보장되지 않음)
- 디바이스 (동질적 하드웨어 필요)

## 실험 결과: 비판적 분석

### 주요 성능 결과

| 구성 | DeepSpeed-Ulysses | Megatron-LM | 속도향상 | 현실 검증 |
|---|---|---|---|---|
| **7B 모델, 8K seq** | 175 TFLOPs | 105 TFLOPs | 1.67x | **좋음**: 견고한 개선 |
| **7B 모델, 32K seq** | 175 TFLOPs | 85 TFLOPs | 2.06x | **주장에 근접** |
| **7B 모델, 128K seq** | 165 TFLOPs | OOM | ∞ | **능력 해제** |
| **30B 모델, 8K seq** | 165 TFLOPs | 45 TFLOPs | 3.67x | **주장 초과** |
| **7B Sparse, 256K seq** | 65 TFLOPs | OOM | ∞ | **성능 붕괴 우려** |

**핵심 통찰:**
- **가변 성능**: 속도향상이 1.35x에서 3.67x까지 다양, 균일한 "2.5x" 주장과 모순
- **능력 vs 성능**: 명확한 능력 해제 (더 긴 시퀀스) vs 혼재된 성능 향상
- **Sparse Attention 문제**: 상당한 성능 저하로 "attention-agnostic" 주장이 과장됨

### 스케일링 분석

| 연구 유형 | 구성 | 효율성 | 해석 |
|---|---|---|---|
| **Strong Scaling** | 131K seq, 64→256 GPUs | 165→136 TFLOPs (18% 손실) | **통신 오버헤드**가 O(N/P) 이론과 모순 |
| **Proportional Scaling** | Seq∝GPUs, 65K→262K | 161→147 TFLOPs (9% 손실) | **더 나은 그러나 완벽하지 않은** 스케일링 |

```python
# 스케일링 결과는 숨겨진 비용을 드러냄:
def real_communication_complexity(N, P, alpha=0.1):
    """
    실제 통신은 오버헤드 요소들을 포함
    """
    theoretical = N / P
    network_contention = alpha * P  # 디바이스 수에 따라 증가
    return theoretical + network_contention

# 이것이 strong scaling 실험에서 효율성 손실을 설명함
```

### 통계적 엄밀성 부족
**중요한 결함**: 오차막대, 신뢰구간, 또는 다중 실행 보고 없음. 성능 주장을 하는 시스템 논문에서 이는 신뢰성을 훼손합니다.

## 강점과 제약

### 주요 강점
1. **근본적 능력 해제**: 이전보다 4배 긴 시퀀스 훈련 가능
2. **품질 보존**: 수렴 연구에서 모델 품질에 영향 없음을 확인
3. **광범위한 적용성**: 다양한 모델 크기에서 작동, 기존 최적화와 통합
4. **이론적 기반**: 시퀀스 병렬처리를 위한 통신 복잡도 분석 제공

### 중요한 제약
1. **성능 불일치**: 구성에 따라 개선이 극적으로 다름
2. **인프라 의존성**: 고대역폭, 저지연 네트워크 필요
3. **Sparse Attention 문제**: 일반성 주장과 모순되는 상당한 성능 저하
4. **실험 엄밀성**: 성능 주장의 통계적 검증 부족

### 숨겨진 복잡성
```python
def performance_prediction(model_size, seq_len, world_size, network_quality):
    """
    성능 향상은 단순한 메트릭으로 예측 불가능
    """
    # 실제 성능에 영향을 주는 요소들:
    communication_ratio = estimate_comm_vs_compute(seq_len, world_size)
    memory_pressure = check_memory_bottlenecks(model_size, seq_len)
    network_efficiency = evaluate_all_to_all_performance(network_quality)
    load_balance = assess_workload_distribution(seq_len, world_size)
    
    # 성능은 모든 요소들의 복잡한 상호작용에 의존
    return complex_interaction(communication_ratio, memory_pressure, 
                             network_efficiency, load_balance)
```

## 실용적 함의

### DeepSpeed-Ulysses 사용 시점
**강력한 후보:**
- 32K 토큰 이상의 시퀀스 훈련
- Dense attention 패턴
- 고대역폭 클러스터 인프라
- 원시 성능보다 능력 해제(더 긴 시퀀스)가 중요한 애플리케이션

**부적합한 후보:**
- 짧은 시퀀스 (< 8K 토큰) - 오버헤드가 지배적
- Sparse attention 패턴 - 성능 저하
- 제한된 네트워크 대역폭 - 통신이 병목
- 지연시간에 민감한 추론 - 이 용도로 설계되지 않음

### 구현 체크리스트
```python
def deployment_readiness_check():
    """
    성공적인 배포를 위한 전제조건
    """
    checks = {
        "sequence_divisibility": sequence_length % world_size == 0,
        "head_divisibility": num_heads % world_size == 0,
        "network_bandwidth": bandwidth > world_size * 10,  # GB/s
        "memory_capacity": can_fit_attention_matrix(),
        "load_balance": validate_uniform_hardware(),
    }
    
    return all(checks.values()), checks
```

## 향후 방향과 연구 기회

### 기술적 개선
1. **적응적 통신**: 네트워크 조건에 기반한 동적 조정
2. **이기종 최적화**: 혼합 하드웨어 환경 지원
3. **Sparse Attention 통합**: 구조화된 sparsity 패턴에 대한 더 나은 지원
4. **메모리 최적화**: all-to-all 연산 중 최대 메모리 감소

### 이론적 확장
```python
# 잠재적 연구 방향:
def future_work_opportunities():
    return [
        "최적 헤드 분산 전략",
        "통신-계산 중첩 기법", 
        "다른 병렬처리 차원과의 통합",
        "실제 네트워크 효과의 이론적 분석",
        "다른 어텐션 메커니즘으로의 확장 (예: cross-attention)"
    ]
```

### 더 넓은 영향
이 연구는 Transformer 훈련 스케일링에 대한 사고방식의 **패러다임 전환**을 나타냅니다:
- **이전**: 배치 크기, 모델 크기, 또는 깊이 스케일링
- **이후**: 시퀀스 길이를 일급 시민으로 스케일링

이는 완전히 새로운 애플리케이션 클래스와 긴 컨텍스트 AI 시스템의 연구 방향을 가능하게 합니다.

## 최종 평가

DeepSpeed-Ulysses는 AI 시스템 환경에서 실제적이고 중요한 문제를 해결하는 **중요한 기여**입니다. 실험 검증에 공백이 있고 성능 주장이 다소 과장되었지만, 핵심 혁신은 건전하고 능력 해제는 진정한 것입니다.

**실무자들을 위해**: 특정 사용 사례(긴 시퀀스, dense attention, 좋은 인프라)에는 가치 있는 도구이지만 범용 솔루션은 아닙니다.

**연구자들을 위해**: 어텐션 중심 병렬처리 접근법은 시스템 최적화에서 새로운 길을 열고 시퀀스 스케일링의 향후 연구를 위한 기반을 제공합니다.

**결론**: 새로운 능력을 가능하게 하는 AI 시스템의 의미 있는 발전이며, 구현 견고성과 실험 엄밀성 모두에서 개선의 여지가 있습니다.

---