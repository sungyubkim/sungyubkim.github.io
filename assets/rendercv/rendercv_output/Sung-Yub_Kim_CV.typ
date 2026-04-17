// Import the rendercv function and all the refactored components
#import "@preview/rendercv:0.3.0": *

// Apply the rendercv template with custom configuration
#show: rendercv.with(
  name: "Sung-Yub Kim",
  title: "Sung-Yub Kim - CV",
  footer: context { [#emph[Sung-Yub Kim -- #str(here().page())\/#str(counter(page).final().first())]] },
  top-note: [ #emph[Last updated in Apr 2026] ],
  locale-catalog-language: "en",
  text-direction: ltr,
  page-size: "us-letter",
  page-top-margin: 0.7in,
  page-bottom-margin: 0.7in,
  page-left-margin: 0.7in,
  page-right-margin: 0.7in,
  page-show-footer: false,
  page-show-top-note: true,
  colors-body: rgb(0, 0, 0),
  colors-name: rgb(0, 0, 0),
  colors-headline: rgb(0, 0, 0),
  colors-connections: rgb(0, 0, 0),
  colors-section-titles: rgb(0, 0, 0),
  colors-links: rgb(0, 0, 0),
  colors-footer: rgb(128, 128, 128),
  colors-top-note: rgb(128, 128, 128),
  typography-line-spacing: 0.6em,
  typography-alignment: "justified",
  typography-date-and-location-column-alignment: right,
  typography-font-family-body: "XCharter",
  typography-font-family-name: "XCharter",
  typography-font-family-headline: "XCharter",
  typography-font-family-connections: "XCharter",
  typography-font-family-section-titles: "XCharter",
  typography-font-size-body: 10pt,
  typography-font-size-name: 25pt,
  typography-font-size-headline: 10pt,
  typography-font-size-connections: 10pt,
  typography-font-size-section-titles: 1.2em,
  typography-small-caps-name: false,
  typography-small-caps-headline: false,
  typography-small-caps-connections: false,
  typography-small-caps-section-titles: false,
  typography-bold-name: false,
  typography-bold-headline: false,
  typography-bold-connections: false,
  typography-bold-section-titles: true,
  links-underline: true,
  links-show-external-link-icon: false,
  header-alignment: center,
  header-photo-width: 3.5cm,
  header-space-below-name: 0.7cm,
  header-space-below-headline: 0.7cm,
  header-space-below-connections: 0.7cm,
  header-connections-hyperlink: true,
  header-connections-show-icons: false,
  header-connections-display-urls-instead-of-usernames: true,
  header-connections-separator: "|",
  header-connections-space-between-connections: 0.5cm,
  section-titles-type: "with_full_line",
  section-titles-line-thickness: 0.5pt,
  section-titles-space-above: 0.5cm,
  section-titles-space-below: 0.3cm,
  sections-allow-page-break: true,
  sections-space-between-text-based-entries: 0.15cm,
  sections-space-between-regular-entries: 0.42cm,
  entries-date-and-location-width: 4.15cm,
  entries-side-space: 0cm,
  entries-space-between-columns: 0.1cm,
  entries-allow-page-break: false,
  entries-short-second-row: false,
  entries-degree-width: 1cm,
  entries-summary-space-left: 0cm,
  entries-summary-space-above: 0.08cm,
  entries-highlights-bullet:  text(13pt, [•], baseline: -0.6pt) ,
  entries-highlights-nested-bullet:  text(13pt, [•], baseline: -0.6pt) ,
  entries-highlights-space-left: 0cm,
  entries-highlights-space-above: 0.08cm,
  entries-highlights-space-between-items: 0.08cm,
  entries-highlights-space-between-bullet-and-text: 0.3em,
  date: datetime(
    year: 2026,
    month: 4,
    day: 17,
  ),
)


= Sung-Yub Kim

#connections(
  [Seoul, Korea],
  [#link("mailto:sungyub.kim@mli.kaist.ac.kr", icon: false, if-underline: false, if-color: false)[sungyub.kim\@mli.kaist.ac.kr]],
  [#link("https://github.com/sungyubkim", icon: false, if-underline: false, if-color: false)[github.com\/sungyubkim]],
  [#link("https://linkedin.com/in/sung-yub-kim-0a82a1264", icon: false, if-underline: false, if-color: false)[linkedin.com\/in\/sung-yub-kim-0a82a1264]],
)


== Education

#education-entry(
  [
    #strong[KAIST], Ph.D. in Graduate School of AI

  ],
  [
    Feb 2019 – Feb 2024

  ],
  main-column-second-row: [
    - #strong[Advisor:] Eunho Yang (Machine Learning & Intelligence Lab.)

    - #strong[Thesis:] Post-hoc analysis techniques and their utilization of pre-trained Neural Networks.

  ],
)

#education-entry(
  [
    #strong[SNU], M.S. in Industrial Engineering

  ],
  [
    Mar 2017 – Feb 2019

  ],
  main-column-second-row: [
    - #strong[Advisor:] Sung-Pil Hong (Management Science & Optimization Lab.)

    - #strong[Thesis:] Train rescheduling via power regret matching algorithm.

  ],
)

#education-entry(
  [
    #strong[SNU], B.S. in Industrial Engineering & Mathematical Science (Double Major)

  ],
  [
    Mar 2013 – Feb 2017

  ],
  main-column-second-row: [
  ],
)

== Experience

#regular-entry(
  [
    #strong[Research Scientist], NAVER Cloud -- Seongnam, Korea

  ],
  [
    Apr 2026 – present

  ],
  main-column-second-row: [
    - Post-Training team \@ HyperScale AI.

    - Stabilized RLVR training of Mixture-of-Experts (MoE) models.

    - Mitigated repetitive generation in LLMs via mechanistic interpretability analysis.

  ],
)

#regular-entry(
  [
    #strong[Staff Engineer], Samsung Electronics -- Suwon, Korea

  ],
  [
    Feb 2024 – Mar 2026

  ],
  main-column-second-row: [
    - Post-hoc training team \@ Language Intelligence PJT., Core Algorithm Lab, AI Center.

    - Improved the Instruction-Following capacity of LLM (≥ 70B) with DPO & Synthetic Data Generation.

    - Improved the reasoning capacity of LLM (≥ 235B) with RLVR & Dataset Pruning.

    - Achieved 49\% on CVDP Spec2RTL benchmark via RL fine-tuning of GPT-Oss (120B), surpassing Claude 3.7 Sonnet (48\%) and DeepSeek-R1 (44\%).

    - Optimized training (FSDPs, NeMo) & inference (vLLM, SGLang) frameworks for LLMs in HPC (≥ 256 GPUs).

  ],
)

== Publications

  #regular-entry(
  [
    #strong[LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding]

  ],
  [
  ],
  main-column-second-row: [
    Doohyuk Jang, Sihwan Park, June Yong Yang, Yeonsung Jung, Jihun Yun, Souvik Kundu, #emph[Sung-Yub Kim], Eunho Yang

    #link("https://openreview.net/pdf?id=98d7DLMGdt")[openreview.net\/pdf?id=98d7DLMGdt] (ICLR 2025)

  ],
)

  #regular-entry(
  [
    #strong[A Simple Remedy for Dataset Bias via Self-Influence: A Mislabeled Sample Perspective]

  ],
  [
  ],
  main-column-second-row: [
    Yeonsung Jung, Jaeyun Song, June Yong Yang, Jin-Hwa Kim, #emph[Sung-Yub Kim], Eunho Yang

    #link("https://openreview.net/pdf?id=ZVrrPNqHFw")[openreview.net\/pdf?id=ZVrrPNqHFw] (NeurIPS 2024)

  ],
)

  #regular-entry(
  [
    #strong[TTD: Text-Tag Self-Distillation Enhancing Image-Text Alignment in CLIP to Alleviate Single Tag Bias]

  ],
  [
  ],
  main-column-second-row: [
    Sanghyun Jo, Soohyun Ryu, #emph[Sung-Yub Kim], Eunho Yang, Kyungsu Kim

    #link("https://arxiv.org/pdf/2404.00384")[arxiv.org\/pdf\/2404.00384] (ECCV 2024)

  ],
)

  #regular-entry(
  [
    #strong[GEX: A flexible method for approximating influence via Geometric Ensemble]

  ],
  [
  ],
  main-column-second-row: [
    #emph[Sung-Yub Kim], Kyungsu Kim, Eunho Yang

    #link("https://openreview.net/pdf?id=tz4ECtAu8e")[openreview.net\/pdf?id=tz4ECtAu8e] (NeurIPS 2023)

  ],
)

  #regular-entry(
  [
    #strong[RGE: A Repulsive Graph Rectification for Node Classification via Influence]

  ],
  [
  ],
  main-column-second-row: [
    Jaeyun Song, #emph[Sung-Yub Kim], Eunho Yang

    #link("https://openreview.net/pdf?id=OcKwZhPwHA")[openreview.net\/pdf?id=OcKwZhPwHA] (ICML 2023)

  ],
)

  #regular-entry(
  [
    #strong[Scale-invariant Bayesian Neural Networks with Connectivity Tangent Kernel]

  ],
  [
  ],
  main-column-second-row: [
    #emph[Sung-Yub Kim], Sihwan Park, Kyungsu Kim, Eunho Yang

    #link("https://openreview.net/pdf?id=OcKwZhPwHA")[openreview.net\/pdf?id=OcKwZhPwHA] (ICLR 2023)

  ],
)

  #regular-entry(
  [
    #strong[Generalized Tsallis Entropy Reinforcement Learning and Its Application to Soft Mobile Robots]

  ],
  [
  ],
  main-column-second-row: [
    Kyungjae Lee, #emph[Sung-Yub Kim], Sungbin Lim, Sungjoon Choi, Mineui Hong, Jae In Kim, Yong-Lae Park, Songhwai Oh

    (RSS 2020)

  ],
)

== Projects

#regular-entry(
  [
    #strong[A machine learning and statistical inference framework for explainable artificial intelligence]

  ],
  [
    Jan 2022 – Dec 2024

  ],
  main-column-second-row: [
    - Institute of Information & Communications Technology Planning & Evaluation (IITP).

    - Topic: Bayesian Neural Networks, Influence Function

  ],
)

#regular-entry(
  [
    #strong[Autonomous Intelligent Digital Companion Framework and Application]

  ],
  [
    Jan 2019 – Dec 2021

  ],
  main-column-second-row: [
    - Institute of Information & Communications Technology Planning & Evaluation (IITP).

    - Topic: Continual Learning, Meta Learning

  ],
)

== Technical Skills

#strong[Coding:] Python, R, LaTeX

#strong[ML Frameworks:] PyTorch, Megatron-LM, verl, vLLM, SGLang

#strong[Distributed Computing:] NVIDIA Certificates:

- #link("https://learn.nvidia.com/certificates?id=Q6pX3qa4SrCjQoKUA5-IZw")[Model Parallelism: Building and Deploying Large Neural Networks]
- #link("https://learn.nvidia.com/certificates?id=_RZWRDEkTwGSXko2GyhCig")[Fundamentals of Accelerated Computing with CUDA Python]

#strong[Research Areas:] Bayesian Neural Networks, Influence Functions, Continual Learning, Meta Learning

#strong[Languages:] Korean, English
