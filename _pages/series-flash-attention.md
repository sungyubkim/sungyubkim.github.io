---
layout: page
title: "FlashAttention Series"
permalink: /blog/series/flash-attention/
description: "IO-aware attention 알고리즘의 발전 과정을 다루는 시리즈입니다. FlashAttention v1부터 v3까지 핵심 아이디어와 최적화 기법을 정리합니다."
nav: false
---

이 시리즈는 **FlashAttention**의 발전 과정을 다룹니다. GPU 메모리 계층 구조를 활용한 IO-aware attention 알고리즘이 어떻게 진화해왔는지 살펴봅니다.

{% assign series_posts = site.posts | where: "series", "flash-attention" | sort: "series_order" %}
<ol class="series-landing-list">
{% for post in series_posts %}
  <li>
    <a href="{{ post.url | relative_url }}"><strong>{{ post.title }}</strong></a>
    <p>{{ post.description }}</p>
  </li>
{% endfor %}
</ol>
