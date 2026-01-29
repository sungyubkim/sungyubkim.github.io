---
layout: page
title: "Sequence Parallelism Series"
permalink: /blog/series/sequence-parallelism/
description: "긴 시퀀스 처리를 위한 Sequence Parallelism 기법을 다루는 시리즈입니다. Ring Attention부터 USP까지 발전 과정을 정리합니다."
nav: false
---

이 시리즈는 **Sequence Parallelism** 기법들을 다룹니다. 긴 시퀀스를 여러 GPU에 분산하여 처리하는 다양한 접근 방식의 발전 과정을 살펴봅니다.

{% assign series_posts = site.posts | where: "series", "sequence-parallelism" | sort: "series_order" %}
<ol class="series-landing-list">
{% for post in series_posts %}
  <li>
    <a href="{{ post.url | relative_url }}"><strong>{{ post.title }}</strong></a>
    <p>{{ post.description }}</p>
  </li>
{% endfor %}
</ol>
