// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "Blog",
          description: "Research notes on ML systems, distributed training, and foundation models.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "Publications",
          description: "Publications in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-flash-attention-3",
        
          title: "Flash Attention 3",
        
        description: "FlashAttention-3 achieves 1.5-2x speedup on H100 GPUs via warp-specialization, WGMMA pipelining, and FP8 support.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/fa3/";
          
        },
      },{id: "post-flash-attention-2",
        
          title: "Flash Attention 2",
        
        description: "FlashAttention-2 doubles training speed through better work partitioning, reduced non-matmul FLOPs, and improved parallelism.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/fa2/";
          
        },
      },{id: "post-flash-attention",
        
          title: "Flash Attention",
        
        description: "FlashAttention: IO-aware attention algorithm achieving 2-3x speedup with 10-20x memory reduction via tiling and online softmax.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/fa/";
          
        },
      },{id: "post-unified-sequence-parallelism",
        
          title: "Unified Sequence Parallelism",
        
        description: "Unified Sequence Parallelism (USP) combines Ulysses and Ring Attention for scalable long-sequence training up to 208K tokens.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/usp/";
          
        },
      },{id: "post-reducing-activation-recomputation-in-large-transformer-models",
        
          title: "Reducing Activation Recomputation in Large Transformer Models",
        
        description: "Techniques for reducing activation recomputation overhead in large Transformer training via sequence and selective checkpointing.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/sp/";
          
        },
      },{id: "post-deepspeed-ulysses",
        
          title: "DeepSpeed Ulysses",
        
        description: "DeepSpeed-Ulysses achieves efficient sequence parallelism via all-to-all communication for 1M+ token training.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/deepspeed_ulysses/";
          
        },
      },{id: "post-blockwise-ringattention",
        
          title: "Blockwise RingAttention",
        
        description: "Blockwise RingAttention solves the memory bottleneck for processing very long sequences by combining blockwise computation with ring communication.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/blockwise_ringattention/";
          
        },
      },{id: "post-mixture-of-experts",
        
          title: "Mixture of Experts",
        
        description: "Sparsely-Gated Mixture-of-Experts (MoE) enables 1000x parameter scaling with minimal compute overhead.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/moe/";
          
        },
      },{id: "post-ring-self-attention",
        
          title: "Ring Self-Attention",
        
        description: "Ring Self-Attention enables sequence parallelism across GPUs using ring communication patterns for distributed attention.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/ring-self-attention/";
          
        },
      },{id: "post-pipeline-parallel-gpipe",
        
          title: "Pipeline Parallel (GPipe)",
        
        description: "Deep dive into GPipe&#39;s micro-batch pipeline parallelism for training models beyond single-device memory limits.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/pp/";
          
        },
      },{id: "post-tensor-parallel",
        
          title: "Tensor Parallel",
        
        description: "Megatron-LM&#39;s tensor model parallelism for training large Transformer models across multiple GPUs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/tp/";
          
        },
      },{id: "post-vim-cheatsheet",
        
          title: "Vim Cheatsheet 📜",
        
        description: "A comprehensive Vim cheatsheet covering navigation, editing, search, and advanced commands.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/vim_cheatsheet/";
          
        },
      },{id: "post-홈페이지-관리",
        
          title: "홈페이지 관리 🏠",
        
        description: "Notes on managing and updating this Jekyll-based GitHub Pages homepage.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/homepage-update/";
          
        },
      },{id: "post-terminologies-in-differential-calculus",
        
          title: "Terminologies in Differential Calculus",
        
        description: "Key terminologies and concepts in differential calculus including pushforward, pullback, and differential forms.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/diffcal/";
          
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%73%75%6E%67%79%75%62.%6B%69%6D@%6D%6C%69.%6B%61%69%73%74.%61%63.%6B%72", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/sungyubkim", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=m2rhgrkAAAAJ", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/sung-yub-kim-0a82a1264", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/SungyubK", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
