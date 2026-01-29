---
title: "Ring Self-Attention"
date: 2025-05-30
last_modified_at: 2025-05-30
layout: post
permalink: /blog/ring-self-attention/
description: "Ring Self-Attention enables sequence parallelism across GPUs using ring communication patterns for distributed attention."
tags: sequence-parallelism ring-attention distributed-training
related_posts: true
disqus_comments: true
giscus_comments: false
toc:
  sidebar: left
---

# TL;DR

> 이 논문은 **시퀀스 병렬화(sequence parallelism)**를 소개합니다 - 전체 시퀀스를 하나의 GPU에 저장해야 하는 기존 방식 대신, 시퀀스를 여러 GPU에 분할하여 더 긴 시퀀스로 Transformer 모델을 훈련하는 새로운 방법입니다. 핵심 혁신은 영리한 링 통신 패턴을 통해 분산 어텐션 계산을 가능하게 하는 **링 셀프 어텐션(Ring Self-Attention, RSA)**입니다.
>
> **주요 기여:**
> - 새로운 병렬화 차원: 시퀀스 길이 (기존 데이터/모델 병렬화와 대비)
> - 분산 어텐션 계산을 위한 링 셀프 어텐션 알고리즘
> - 텐서 병렬화 대비 3배 긴 시퀀스와 13.7배 큰 배치 크기

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
- 기존 병렬화 전략과 호환 가능하여 "4차원 병렬화" 구현

**비판적 현실 점검:** 실험 검증에 상당한 약점이 있습니다 - 불공정한 기준선, 제한된 범위, 과장된 개선 효과. 이 방법은 일반적인 돌파구라기보다는 특정 영역(많은 GPU에서 중간-긴 시퀀스)을 위한 것입니다.

# Takeaways

## 문제와 동기

### 왜 중요한가
```python
# 기존 제약: 전체 시퀀스가 하나의 GPU에 맞아야 함
traditional_limit = gpu_memory / (sequence_length^2 * batch_size * hidden_size)

# 의료 영상 예시
medical_image = 512 * 512 * 512  # 3D 의료 스캔
tokens_per_image = 134_million    # 각 복셀이 토큰이 됨
memory_required = "단일 GPU로는 불가능"

# 이것이 시퀀스 병렬화가 해결하는 근본적인 문제
```

**핵심 통찰**: 기존 연구가 알고리즘적 개선(스파스 어텐션, 선형 어텐션)에 집중하는 반면, 이 논문은 **시스템 접근법**을 취합니다 - 어텐션을 더 효율적으로 만드는 대신 더 많은 GPU를 사용하여 긴 시퀀스를 처리합니다.

### 현재 병렬화 환경
```python
parallelism_strategies = {
    'data_parallelism': '배치를 GPU에 분할',
    'pipeline_parallelism': '모델 레이어를 GPU에 분할', 
    'tensor_parallelism': '모델 매개변수를 GPU에 분할',
    'sequence_parallelism': '시퀀스 길이를 GPU에 분할 (새로운 방식)'
}
```

## 기술적 혁신: 링 셀프 어텐션

### 핵심 알고리즘 설명

**1단계: 어텐션 스코어 QK^T 계산**
```python
def ring_attention_scores_example():
    # 설정: 4개 GPU, 시퀀스 길이 8, 각 GPU가 2개 토큰 처리
    devices = {
        0: "토큰 [0,1]",
        1: "토큰 [2,3]", 
        2: "토큰 [4,5]",
        3: "토큰 [6,7]"
    }
    
    # Device 1의 관점에서 살펴보기
    device_1_queries = "토큰 [2,3]에 대한 Q"
    
    # Device 1은 모든 키와 어텐션을 계산해야 함
    ring_steps = [
        "단계 0: 로컬 K[2,3] 사용 -> Q[2,3] @ K[2,3]^T 계산",
        "단계 1: Device 0에서 K[0,1] 받음 -> Q[2,3] @ K[0,1]^T", 
        "단계 2: Device 3에서 K[6,7] 받음 -> Q[2,3] @ K[6,7]^T",
        "단계 3: Device 2에서 K[4,5] 받음 -> Q[2,3] @ K[4,5]^T"
    ]
    
    # 결과: Device 1은 토큰 [2,3]이 모든 토큰 [0-7]에 어텐션하는 스코어를 가짐
    attention_matrix_shape = "[배치, 2_로컬_토큰, 8_전체_토큰]"
    
    return f"Device 1이 완전한 어텐션 스코어 계산: {attention_matrix_shape}"
```

**2단계: 값을 사용한 최종 출력 계산**
```python
def ring_attention_output_example():
    # Device 1은 어텐션 가중치를 가짐: [배치, 2, 8] 
    # 이제 모든 값 임베딩과 곱해야 함
    
    ring_steps = [
        "단계 0: 로컬 V[2,3] 사용 -> weights[:,:,2:4] @ V[2,3]",
        "단계 1: V[0,1] 받음 -> weights[:,:,0:2] @ V[0,1]",
        "단계 2: V[6,7] 받음 -> weights[:,:,6:8] @ V[6,7]", 
        "단계 3: V[4,5] 받음 -> weights[:,:,4:6] @ V[4,5]"
    ]
    
    final_output = "모든 부분 출력의 합 = 토큰 [2,3]에 대한 완전한 어텐션"
    return final_output
```

### 중요한 구현 요구사항

**메모리 관리**
```python
def memory_requirements():
    # 중요: 어텐션 행렬은 여전히 전체 시퀀스 차원이 필요
    peak_memory_components = {
        'attention_scores': 'batch_size × local_seq_len × 전체_seq_len',
        'attention_weights': 'batch_size × local_seq_len × 전체_seq_len', 
        'temp_embeddings': '2 × batch_size × local_seq_len × hidden_size',
        'gradients': '역전파 중 추가 2배'
    }
    
    # 경고: 매우 긴 시퀀스의 경우 어텐션 행렬이 여전히 OOM 유발 가능
    memory_scaling = "O(sequence_length^2) 대신 O(sequence_length), 하지만 주의사항 있음"
    return memory_scaling
```

**동기화 요구사항**
```python
def synchronization_constraints():
    assumptions = [
        "모든 장치가 링 통신에 참여해야 함",
        "링 통신은 동기식이어야 함", 
        "어떤 장치 실패든 전체 링을 중단시킴",
        "시퀀스 길이가 장치 수로 나누어떨어져야 함",
        "모든 장치가 동일한 로컬 계산 시간을 가져야 함"
    ]
    
    failure_modes = [
        "장치 실패 -> 전체 훈련 중단",
        "네트워크 분할 -> 링 통신 실패",
        "로드 불균형 -> 가장 느린 장치가 속도 결정",
        "메모리 파편화 -> 일관성 없는 성능"
    ]
    
    return assumptions, failure_modes
```

## 실험 결과 분석

### 주요 결과와 비판적 해석

| 실험 | 기준선 | 시퀀스 병렬화 | 주장된 개선 | **현실 점검** |
|------|--------|---------------|------------|---------------|
| **최대 배치 크기** | 116 (12 GPUs) | 1,590 (64 GPUs) | 13.7× | **불공정**: 5배 많은 GPU 사용. 실제 알고리즘 개선은 ~2-3× |
| **최대 시퀀스 길이** | 750 | 2,250 | 3.0× | **보통**: 합리적 개선이지만 혁명적이지 않음 |
| **처리량** | 18K tokens/sec | 24K tokens/sec | 1.3× | **미미함**: 통신 오버헤드를 겨우 보상 |
| **스파스 어텐션** | 4.2K tokens | 114K tokens | 27× | **오해의 소지**: 분산 vs 단일장치 비교, 알고리즘 비교 아님 |

### 부족한 절제 연구

| 구성요소 | 테스트된 것 | **누락된 것** |
|----------|-------------|---------------|
| 링 통신 | 기본 기능 | all-gather, all-reduce 대안과 **비교 없음** |
| 2단계 설계 | 정확성 검증 | 단일 단계 또는 다른 분해의 **분석 없음** |
| 통신 패턴 | 링 토폴로지 | 트리, 버터플라이, 메시 패턴 **평가 없음** |
| 메모리 vs 통신 | 정성적 논의 | **체계적** 트레이드오프 분석 없음 |

## 비판적 평가

### 강점
1. **새로운 문제 프레이밍**: 시퀀스 길이를 병렬화 차원으로 체계적으로 다룬 첫 연구
2. **견고한 기술적 혁신**: 링 셀프 어텐션이 수학적으로 정확하고 우아함
3. **구현 가능성**: 순수 PyTorch, 특별한 컴파일러 불필요
4. **호환성**: 기존 병렬화 전략과 함께 작동

### 주요 한계점

**실험 검증 문제**
```python
experimental_problems = {
    'baseline_bias': '시퀀스 스케일링에 텐서 병렬화와 비교',
    'missing_baselines': '그래디언트 체크포인팅, 혼합 정밀도와 비교 없음',
    'hardware_dated': '2016년 P100 GPU에서만 테스트',
    'limited_scope': 'Wikipedia의 BERT만, 다양한 워크로드 없음',
    'statistical_rigor': '오차 막대, 분산 분석, 유의성 테스트 없음'
}
```

**실제 배포 과제**
```python
deployment_reality = {
    'hardware_requirements': '고대역폭 인터커넥트(InfiniBand) 필수',
    'optimal_scale': '16-32 GPU (이후 통신 오버헤드)',
    'sweet_spot': '중간 시퀀스 (1K-4K 토큰), 짧거나 매우 긴 것 아님',
    'fault_tolerance': '단일 장치 실패가 전체 훈련 중단',
    'complexity': '대안 대비 상당한 구현 복잡성'
}
```

## 실용적 함의

### 시퀀스 병렬화를 언제 사용할지
```python
def should_use_sequence_parallelism(use_case):
    good_fit = [
        "4K+ 토큰 시퀀스를 가진 의료 영상",
        "긴 문서 처리 (법률, 과학 논문)",
        "유전체 시퀀스 분석",
        "비전 트랜스포머를 사용한 고해상도 이미지 분석"
    ]
    
    poor_fit = [
        "표준 NLP 작업 (≤512 토큰)",
        "일반적인 이미지 크기의 컴퓨터 비전", 
        "자원 제약 환경",
        "높은 신뢰성이 필요한 프로덕션 시스템"
    ]
    
    considerations = {
        'minimum_gpus': 4,  # 이하에서는 오버헤드가 지배적
        'network_requirements': '고대역폭, 저지연 인터커넥트',
        'sequence_length': '최적 지점: 1K-8K 토큰',
        'batch_size': '큰 배치가 이익 증폭'
    }
    
    return good_fit, poor_fit, considerations
```

### 구현 가이드
```python
class SequenceParallelismChecklist:
    prerequisites = [
        "고대역폭 네트워크 (InfiniBand 권장)",
        "동일한 하드웨어 (같은 GPU)",
        "GPU 수로 나누어떨어지는 시퀀스 길이",
        "견고한 분산 훈련 인프라"
    ]
    
    optimization_tips = [
        "링 통신 버퍼 미리 할당",
        "가능한 곳에 비동기 통신 사용", 
        "메모리용 그래디언트 체크포인팅 구현",
        "장치 간 로드 불균형 모니터링",
        "장치 실패에 대한 대안 전략 보유"
    ]
    
    performance_tuning = {
        'batch_size': '메모리 한계까지 증가',
        'sequence_chunks': '계산 vs 통신 균형',
        'ring_ordering': '네트워크 토폴로지에 최적화',
        'memory_management': '버퍼 풀로 파편화 방지'
    }
```

## 향후 연구 방향

### 즉각적인 개선 필요사항
1. **엄격한 실험 검증**: 공정한 기준선, 최신 하드웨어, 다양한 워크로드
2. **장애 내성**: 장치 실패를 우아하게 처리
3. **동적 로드 밸런싱**: 이질적 하드웨어에 적응
4. **통신 최적화**: 단순한 링 패턴을 넘어서

### 장기 연구 질문
```python
future_directions = {
    'algorithmic_fusion': '스파스 어텐션, 선형 어텐션과 결합',
    'adaptive_partitioning': '어텐션 패턴 기반 동적 시퀀스 분할',
    'cross_modal_scaling': '멀티모달 트랜스포머로 확장', 
    'efficiency_optimization': '통신 오버헤드 감소',
    'theoretical_analysis': '형식적 복잡도 분석과 최적성 경계'
}
```

### 최신 기법과의 통합
이 논문은 많은 중요한 발전 이전에 나왔습니다:
- **Zero Redundancy Optimizer (ZeRO)**: 메모리 최적화를 위해 결합 가능
- **그래디언트 체크포인팅**: 실용적 배포에 필수
- **혼합 정밀도 훈련**: 현재 표준 관행
- **효율적 어텐션 변형**: FlashAttention, LinearAttention 등

## 결론

**시퀀스 병렬화는 병렬화 환경에서 특정 틈새를 채우는 견고한 기술적 기여를 나타냅니다.** 링 셀프 어텐션 알고리즘은 우아하고 수학적으로 건전합니다. 그러나 실용적 영향은 논문이 주장하는 것보다 제한적입니다.

**실무자들을 위해**: 중간에서 긴 시퀀스(1K-8K 토큰), 충분한 하드웨어(고대역폭 네트워킹을 가진 16+ GPU), 그리고 시퀀스 길이가 주요 병목인 워크로드가 있을 때 시퀀스 병렬화를 고려하세요.

**연구자들을 위해**: 이 연구는 시퀀스 길이를 스케일링의 새로운 차원으로 열어주지만, 그래디언트 체크포인팅이나 더 효율적인 어텐션 메커니즘 같은 간단한 대안과 실질적으로 경쟁하려면 상당한 작업이 남아있습니다.

근본적인 통찰 - 시스템 수준 솔루션이 알고리즘적 개선을 보완할 수 있다는 것 - 은 가치 있으며 분산 딥러닝의 향후 연구에 영감을 줄 가능성이 높습니다.

---