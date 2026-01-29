---
title: "홈페이지 관리 🏠"
date: 2024-02-21
last_modified_at: 2024-02-21
layout: post
permalink: /blog/homepage-update/
description: "Notes on managing and updating this Jekyll-based GitHub Pages homepage."
tags: github-pages
related_posts: true
disqus_comments: true
giscus_comments: false
---

출처: <https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll>

* 코드 수정한 다음에
```bash
$ bundle install
```

* 수정한 코드를 확인하려면
```bash
$ bundle exec jekyll serve
> Configuration file: /Users/octocat/my-site/_config.yml
>            Source: /Users/octocat/my-site
>       Destination: /Users/octocat/my-site/_site
> Incremental build: disabled. Enable with --incremental
>      Generating...
>                    done in 0.309 seconds.
> Auto-regeneration: enabled for '/Users/octocat/my-site'
> Configuration file: /Users/octocat/my-site/_config.yml
>    Server address: http://127.0.0.1:4000/
>  Server running... press ctrl-c to stop.
```