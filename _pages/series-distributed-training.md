---
layout: page
title: "Distributed Training Series"
permalink: /blog/series/distributed-training/
description: "대규모 모델 학습을 위한 분산 훈련 기법을 다루는 시리즈입니다. Tensor Parallelism, Pipeline Parallelism, Activation Recomputation 등을 정리합니다."
nav: false
---

이 시리즈는 **Distributed Training** 기법들을 다룹니다. 단일 GPU의 한계를 넘어 대규모 Transformer 모델을 효율적으로 훈련하는 병렬화 전략을 살펴봅니다.

{% assign series_posts = site.posts | where: "series", "distributed-training" | sort: "series_order" %}
<ol class="series-landing-list">
{% for post in series_posts %}
  <li>
    <a href="{{ post.url | relative_url }}"><strong>{{ post.title }}</strong></a>
    <p>{{ post.description }}</p>
  </li>
{% endfor %}
</ol>
