---
layout: blog
title: "The Ultra-Long Context Paradox"
date: 2026-05-22
description: "An opinion piece on why we need 1M+ context — not despite its problems, but because of them."
header_image: /assets/blog/ultra-long-context-paradox/title_card.svg
---

*An opinion piece on why we need 1M+ context — not despite its problems, but because of them.*

Context length has grown from 4K tokens to 256K in just two years. Gemini, Claude, GPT — all frontier models now advertise six-figure context lengths. By many accounts, long context is a solved problem.

**It isn't.**

Users and builders are hitting walls. Models degrade on long inputs. Agents lose their thread halfway through complex tasks. The promise of *just paste everything in* crumbles under scrutiny. The field has responded with **context management** — compaction, RAG, sub-agent architectures — designed to keep the effective context small and high-signal. So a natural question emerges: should we keep pushing toward ultra-long context — 1M tokens and beyond — or should we stop here and manage what we have?

I want to share an underappreciated perspective. The answer is paradoxical: the very problems that make long context unreliable are the same problems that demand we scale it further. And the resolution may point toward something bigger than either mentioned approach alone.

## The Problems Are Real

### Context Rot

The most well-documented failure is **context rot**: retrieval accuracy degrades when relevant information sits in the middle of a long context. Liu et al. <sup>[1](#ref-1)</sup> established the *lost-in-the-middle* effect — models retrieve well from the beginning or end, but accuracy drops sharply in the middle.

The root cause is architectural. Attention produces n² pairwise relationships. As Anthropic describes it <sup>[2](#ref-2)</sup>, each model has a finite *attention budget* that depletes with every token:

> a performance gradient rather than a hard cliff: models remain capable but show reduced precision for information retrieval and long-range reasoning.

The key word is *precision*. Context rot degrades the model's ability to locate and connect specific pieces of information across distance — a retrieval problem tied to position.

<p align="center">
  <img src="../assets/blog/ultra-long-context-paradox/fig1_context_rot.svg" alt="Figure 1: Context Rot — The Lost-in-the-Middle Effect" width="680">
</p>
<p align="center"><sub><em>Figure 1. Models attend well at the edges but lose signal in the middle — a U-shaped accuracy curve across context depth.</em></sub></p>

### Context Anxiety

Distinct from context rot — which is a *positional* bias — is what I'd call **context anxiety**: a *behavioral* degradation that emerges as models sense their context limits approaching.

Anthropic documented this in their work on managed agents <sup>[4](#ref-4)</sup>: Claude Sonnet 4.5 would

> wrap up tasks prematurely as it sensed its context limit approaching.

The model didn't fail on a specific task; it became *conservative* — rushing conclusions, narrowing exploration, losing ambition. Levy et al. <sup>[5](#ref-5)</sup> provide broader grounding: reasoning performance degrades as input length grows, even when the extra tokens are irrelevant to the task. The degradation is not about position but *load* — instruction-following decay, coherence loss, style drift.

Context anxiety is more insidious than context rot because it affects the model's *agency and judgment*, not just factual retrieval. A model that can't find a fact gives a wrong answer. A model that prematurely abandons a debugging session gives up on the right answer entirely.

<p align="center">
  <img src="../assets/blog/ultra-long-context-paradox/fig2_rot_vs_anxiety.svg" alt="Figure 2: Context Rot vs. Context Anxiety" width="680">
</p>
<p align="center"><sub><em>Figure 2. Context rot is a positional bias (middle is worse); context anxiety is a behavioral degradation (overall quality fades as context fills).</em></sub></p>

## Context Management: The Pragmatic Answer (and Its Fatal Flaw)

Given these problems, context management seems like the obvious solution. Don't fight the limits — work within them. Anthropic advocates finding *the smallest possible set of high-signal tokens* <sup>[2](#ref-2)</sup>. The toolkit is mature: compaction <sup>[2](#ref-2), [4](#ref-4)</sup>, sub-agent architectures <sup>[2](#ref-2)</sup>, RAG <sup>[6](#ref-6), [7](#ref-7)</sup>, just-in-time retrieval <sup>[2](#ref-2)</sup>. And they work remarkably well — a 16K RAG pipeline often outperforms a raw 128K context on retrieval tasks <sup>[7](#ref-7)</sup>.

**But there's a crack in this argument that grows wider the longer you look at it.**

Anthropic themselves identify the fundamental tension <sup>[4](#ref-4)</sup>: all context management involves **irreversible decisions** about what to retain.

> It is difficult to know which tokens the future turns will need.

Mitigations exist. Anthropic's managed agents store compacted messages externally and maintain cross-session memory so context remains *recoverable* <sup>[4](#ref-4)</sup>. Zhang et al. go further, treating the context as a programmable object the model can recursively examine and slice via code <sup>[31](#ref-31)</sup>. But "recoverable" is not "present." A model that must re-fetch discarded context pays a cost in coherence and reasoning continuity — it has to *know* something is missing before it can look for it. And the *reasoning trace* — why it chose a design, which alternatives it rejected — lives only in the context and vanishes when compressed.

Each compression step is lossy, and each makes the surviving context harder to navigate — compressed summaries lack the structure and specificity needed for precise retrieval, so the model increasingly struggles to find what it needs in its own history. Compress once, you probably kept the important parts. Compress eight times over four hours, and you're reasoning on the ghost of a signal. **Can you summarize your way through an 11-hour coding session?** Each compression bets you know what the future will need. Over enough bets, some will be wrong — and a single wrong bet can cascade into failure.

<p align="center">
  <img src="../assets/blog/ultra-long-context-paradox/fig3_compression_loss.svg" alt="Figure 3: Cumulative Compression Loss" width="680">
</p>
<p align="center"><sub><em>Figure 3. Each compression round is lossy; after 8 rounds over a 4-hour session, the model reasons on a fraction of the original signal.</em></sub></p>

## The Turn: Why Ultra-Long Context Is Non-Negotiable

Here's where the argument inverts. The lossy nature of context management is not an argument *against* long context — it's the strongest argument *for scaling it further*.

Ultra-long context reduces the *frequency* of compression needed. A 1M-token window may carry a four-hour agent session without compressing at all. A 128K window forces compression every thirty minutes — eight lossy rounds, each discarding signal you might need later. **Ultra-long context doesn't eliminate context management. It reduces the damage management inflicts.**

This isn't hypothetical. MiMo-V2.5-Pro <sup>[8](#ref-8)</sup> — with a native 1M-token context and hybrid sparse/global attention — recently demonstrated sustained agency at a scale impossible with aggressive compression:

> **PKU Compiler Project.** A complete SysY compiler in Rust (lexer → parser → Koopa IR → RISC-V backend), **672 tool calls over 4.3 hours**, scoring 233/233 on hidden tests. A task that typically takes a PKU CS major several weeks.

> **Video Editor.** A full desktop video editor — multi-track timeline, cross-fades, audio mixing, export — **8,192 lines of code, 1,868 tool calls, 11.5 hours** of autonomous work.

Similarly, GLM-5.1 <sup>[9](#ref-9)</sup> built a complete Linux desktop environment in a browser — window manager, status bar, applications, VPN, Chinese font support — over **8 hours and 1,200+ steps**, generating 4.8MB of supporting files. In a separate task, it optimized a vector database from 3,108 QPS to 21,472 QPS (**6.9x improvement**) across 655 iterations, autonomously progressing through full scan → IVF bucketing → quantized coarse ranking → early pruning. The authors explicitly identify *context anxiety* as a core technical challenge for this class of sustained work.

A conventional objection: long-context training improves retrieval and reasoning over large inputs, sure — but does it help with the *agency* these sessions demand? The two are distinct: **long-context understanding** (retrieve, synthesize, reason over a given input) vs. **long-horizon agency** (plan, backtrack, recover from errors over many steps). Conventional wisdom treats them as separate axes. But emerging evidence suggests what I'd call **context grokking**: reasoning gains from context scaling stay flat through 4K → 256K, then suddenly manifest past ~512K. Models trained with extended context show improvements not only on long-context benchmarks like LongBench v2 but also on reasoning-intensive tasks like AIME — as if the capacity to hold more state and the capacity to reason over it were latent all along, waiting for a scale threshold to unlock them.

The model must hold the evolving state of a complex system — code it wrote, tests it ran, regressions it diagnosed — across thousands of interactions. MiMo reported diagnosing a regression where *a refactoring pass broke two tests* and recovering, possible only because it still remembered what it had changed and why. The model creators describe *harness awareness*:

> makes full use of the affordances of its harness environment, manages its memory, and shapes how its own context is populated.

Ultra-long context *plus* intelligent management — not one substituting for the other.

**On the RAG finding**: yes, a 16K RAG pipeline beats 128K raw context on *retrieval*. But retrieval asks *where is the answer?* Long-horizon agency asks *given everything I've built, tested, and broken over four hours, what next?* The latter requires uncompressed state — because the subtle dependencies between decisions made hours apart are exactly what compression discards.

Ultra-long context determines the *floor* of what a system can do. Context management raises the *ceiling*. The floor matters more — you can't manage context you never had.

## The Three Pillars: Scaling Without Making Things Worse

If ultra-long context is essential, how do we get there without amplifying context rot, worsening context anxiety, and breaking the bank? Three pillars must be solved simultaneously.

### Pillar 1: Training — Context Parallelism at Scale

A 1M-token sequence doesn't fit in a single device. You need **context parallelism (CP)** — sharding the sequence across devices. The evolution of CP tells a story of increasing sophistication:

**Ring Attention** <sup>[10](#ref-10)</sup> pioneered the idea: pass KV blocks around a ring, computing local attention at each step. It works, but point-to-point communication creates overhead at large CP degrees. **DeepSpeed Ulysses** <sup>[11](#ref-11)</sup> introduced all-to-all CP: partition the sequence, then use all-to-all collectives to transform from sequence-partitioned to head-partitioned layouts. Better bandwidth utilization, but CP degree is bounded by the number of KV heads — a hard ceiling in GQA models. **Llama 3** <sup>[12](#ref-12)</sup> took the all-gather approach in their 4D parallelism: each GPU holds a sequence chunk, all-gather collects full KV tensors so each device computes full causal attention — trading memory for compute-communication overlap.

The field is converging on **hybrid approaches** — all-to-all within nodes (head-parallel over NVLink) with point-to-point across nodes (sequence-parallel over InfiniBand) — breaking the KV-head ceiling while avoiding ring overhead <sup>[13](#ref-13), [14](#ref-14)</sup>.

<p align="center">
  <img src="../assets/blog/ultra-long-context-paradox/fig4_cp_evolution.svg" alt="Figure 4: Evolution of Context Parallelism" width="680">
</p>
<p align="center"><sub><em>Figure 4. Context parallelism has evolved from point-to-point ring exchange to all-to-all and all-gather collectives, converging on hybrid approaches that combine intra-node and inter-node strategies.</em></sub></p>

The cost is real: CP degree grows with context (16x, 32x+), consuming parallelism budget. But I'll return to why cost isn't the argument-ender critics claim.

### Pillar 2: Inference — Memory and Compute Paradigms

At 1M tokens, KV cache alone can consume hundreds of gigabytes per request, making dense attention economically infeasible. The solutions form a spectrum:

**KV compression** at the embedding level: DeepSeek's MLA <sup>[15](#ref-15)</sup> projects keys and values into a lower-dimensional latent space — smaller per-token footprint, but still O(n) in sequence length. MLA replicates KV across DP ranks, so CP becomes necessary for ultra-long inference — where training-time and inference-time parallelism challenges converge <sup>[16](#ref-16)</sup>.

**Semantic compression** at the token level: KSA <sup>[17](#ref-17)</sup> compresses historical contexts into learnable summary tokens at ratio k, achieving O(n/k) cache complexity — building on earlier ideas like gist tokens <sup>[18](#ref-18)</sup> and Activation Beacon <sup>[19](#ref-19)</sup>, but integrating learnable compression into pre-training at the architectural level.

**Sparse attention** at the selection level: DSA in V3.2 <sup>[20](#ref-20)</sup> evolving into CSA in V4 <sup>[21](#ref-21)</sup> — a lightweight indexer scores compressed keys, selects top-k per query, sparse kernel reads only selected entries. Sparse attention doesn't directly reduce KV cache (the full cache must exist for prefix/radix caching), but *enables* offloading to host memory. CSA, operating on compressed keys, admits faster offloading than DSA — potentially enabling SSD-tier transfer, bounding context by storage rather than HBM.

V4 <sup>[21](#ref-21)</sup> also introduces **Highly Compressed Attention (HCA)** and interleaves HCA and CSA in a hybrid architecture. This hybrid pattern is a trend: MiMo-V2.5-Pro uses sliding window and global attention at 6:1; HySparse <sup>[22](#ref-22)</sup> interleaves full and sparse layers, using full attention as an oracle for sparse token selection, achieving ~10x KV-cache reduction in an 80B MoE model. Full disclosure: LoZA <sup>[23](#ref-23)</sup> is my own work, so take this with appropriate salt — it retrofits structured sparse patterns onto existing full-attention models during continued pre-training, enabling 1M-token processing without retraining from scratch. I'm biased, but it fills a gap: the methods above either require baking sparsity into the architecture from day one or involve fairly intricate indexing and selection machinery.


<p align="center">
  <img src="../assets/blog/ultra-long-context-paradox/fig5_inference_spectrum.svg" alt="Figure 5: The Inference Memory Spectrum" width="680">
</p>
<p align="center"><sub><em>Figure 5. The inference memory spectrum ranges from full KV cache (MLA) through semantic compression (KSA) to sparse selection with offloading (DSA/CSA). Hybrid architectures interleave these layers.</em></sub></p>

### Pillar 3: Data — The Scarcity Problem

Where do you find high-quality 1M-token training sequences? You mostly don't. Books average 50K–100K tokens, papers 5K–15K, and code repositories — while large — have sparse long-range dependencies beyond 512K. Concatenating files doesn't teach long-range reasoning.

The field initially relied on **RoPE extension methods** — YaRN <sup>[24](#ref-24)</sup>, LongRoPE <sup>[25](#ref-25)</sup> — to extrapolate short-context models without proportionally long training data. These work, but emerging evidence favors **direct training on genuinely long data**. Fu et al. <sup>[26](#ref-26)</sup> show that 500M–5B well-curated long tokens suffice for 128K context; Gao et al. <sup>[27](#ref-27)</sup> push further — their ProLong-8B achieves 512K capability using only 5% of Llama-3.1's long-context training tokens, by training *beyond* the evaluation length and mixing code with books. The bottleneck has shifted from architecture tricks to data engineering.

The data sources: **curated long-form text** (books, legal filings, multi-document clusters, code with inter-file dependencies); **synthetic generation** — paraphrasing <sup>[28](#ref-28)</sup> that teaches retrieval through rephrasing, and megadocuments <sup>[29](#ref-29)</sup> that stitch sources into long sequences with controlled dependencies (1.8x data efficiency); and **trajectory data** from downstream applications — agent traces, multi-turn tool-use sessions, long-horizon coding sessions. The 672-turn compiler and 1,868-turn video editor sessions from MiMo are exactly the data that would teach sustained long-range reasoning.

The **impossible triangle** looms: quality, diversity, and length. Any two are straightforward; all three at 1M scale remains a data engineering challenge rarely discussed in architecture papers.

<p align="center">
  <img src="../assets/blog/ultra-long-context-paradox/fig6_impossible_triangle.svg" alt="Figure 6: The Impossible Triangle of Long-Context Data" width="680">
</p>
<p align="center"><sub><em>Figure 6. The impossible triangle: quality, diversity, and length are easy to get in pairs, but achieving all three at 1M-token scale remains a data engineering challenge.</em></sub></p>

## The Deeper Frame: Memory Scaling

Zoom out. The deeper concept is **memory scaling**.

| | **Parametric Memory** | **Non-Parametric Memory** |
|---|---|---|
| **What** | Weights learned during training | Context at inference time |
| **Contains** | What the model *knows* | What the model can *see* |
| **Scaling challenge** | More parameters → more compute | More tokens → more attention + KV cache |
| **Sparsity solution** | MoE — activate relevant experts | Sparse attention — attend to relevant tokens |

Long-context efficiency is really about **non-parametric memory scaling**. MoE models do the same for parametric memory — scaling total parameters while keeping active parameters manageable through sparse activation. The unifying insight: **sparsity** — you don't need to activate everything all the time. Sparse attention and sparse parameters are different faces of the same principle.

A **third axis** is emerging. Cheng et al. <sup>[30](#ref-30)</sup> introduce *conditional memory* (Engram) — O(1) lookup that separates knowledge retrieval from neural computation entirely. Scaled to 27B parameters with deterministic addressing and host-memory prefetching, it represents a dimension where memory capacity expands via tiered storage without proportional compute cost, governed by a *U-shaped scaling law* balancing neural computation and static memory.

If AGI requires vast knowledge (parametric), vast working context (non-parametric), and vast retrievable memory (conditional) — then efficient scaling of all three through sparsity and tiered storage is not a luxury. It's the path.

## The Resolution: Neither Alone, Both Together

The paradox, then, has a clean resolution.

Ultra-long context is the **atomic ability** — the floor. How long a horizon the system can sustain, how complex a state it can hold. Without it, no amount of context management substitutes for information that was never there.

Context management is the **intelligence layer** — the amplifier. What to attend to, what to compress, when to retrieve. Without it, even 1M tokens becomes a haystack with no needle-finding strategy.

We see this synthesis already. MiMo-V2.5-Pro's 11.5-hour session combines 1M native context with *harness awareness* — the model actively managing its own memory. GLM-5.1's 8-hour build demonstrates the same: a single model self-iterating through thousands of cycles, inspecting its output, continuing autonomously. Neither raw context nor management alone would have sufficed.

The analogy: working memory capacity vs. cognitive strategy. A person with poor working memory can't compensate with note-taking alone. Good working memory without strategy is also limited. Intelligence emerges from both.

A final word on cost. The three pillars are real engineering challenges. But five years ago, training trillion-parameter models seemed equally prohibitive. The history of deep learning is costs that appeared insurmountable becoming routine through algorithmic innovation, hardware scaling, and engineering will. Dismissing ultra-long context on cost alone is betting against the strongest trend in the field.

The question is not *whether* to build longer context or smarter management. It's that we cannot build the latter without first solving the former. And when we solve both, what emerges may be something qualitatively new.

## References

<div style="font-size: 0.85em; line-height: 1.6;" markdown="1">

<a id="ref-1"></a>**[1]** Nelson F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024.

<a id="ref-2"></a>**[2]** Anthropic. *Effective Context Engineering for AI Agents.* [anthropic.com/engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents), 2025.

<a id="ref-3"></a>**[3]** Junhao Hu et al. *Lil: Less is Less When Applying Post-Training Sparse-Attention Algorithms in Long-Decode Stage.* [arXiv:2601.03043](https://arxiv.org/abs/2601.03043), 2026.

<a id="ref-4"></a>**[4]** Anthropic. *Managed Agents.* [anthropic.com/engineering](https://www.anthropic.com/engineering/managed-agents), 2025.

<a id="ref-5"></a>**[5]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models.* [arXiv:2402.14848](https://arxiv.org/abs/2402.14848), 2024.

<a id="ref-6"></a>**[6]** Patrick Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS, 2020.

<a id="ref-7"></a>**[7]** Xinze Li et al. *Long Context vs. RAG for LLMs: An Evaluation and Revisits.* arXiv, 2025.

<a id="ref-8"></a>**[8]** Xiaomi. *MiMo-V2.5-Pro.* [mimo.xiaomi.com](https://mimo.xiaomi.com/mimo-v2-5-pro), 2026.

<a id="ref-9"></a>**[9]** Zhipu. *GLM-5.1: Building a Linux Desktop from Scratch in 8 Hours.* [Zhihu](https://zhuanlan.zhihu.com/p/2025165819143812557), 2026.

<a id="ref-10"></a>**[10]** Hao Liu, Matei Zaharia, Pieter Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* ICLR, 2024.

<a id="ref-11"></a>**[11]** Sam Ade Jacobs et al. *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models.* [arXiv:2309.14509](https://arxiv.org/abs/2309.14509), 2023.

<a id="ref-12"></a>**[12]** Aaron Grattafiori et al. *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783), 2024.

<a id="ref-13"></a>**[13]** NVIDIA. *Hybrid / Hierarchical Context Parallel.* [NeMo Megatron Bridge Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/hybrid-context-parallel.html), 2025.

<a id="ref-14"></a>**[14]** Ascend. *Hybrid Context Parallel.* [MindSpeed Docs](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/hybrid-context-parallel.md), 2025.

<a id="ref-15"></a>**[15]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), 2024.

<a id="ref-16"></a>**[16]** vLLM Project. *Distributed CP for MLA.* [GitHub PR #23734](https://github.com/vllm-project/vllm/pull/23734), 2025.

<a id="ref-17"></a>**[17]** Chenglong Chu et al. *Kwai Summary Attention Technical Report.* [arXiv:2604.24432](https://arxiv.org/abs/2604.24432), 2026.

<a id="ref-18"></a>**[18]** Jesse Mu, Xiang Lisa Li, Noah Goodman. *Learning to Compress Prompts with Gist Tokens.* arXiv:2304.08467, 2023.

<a id="ref-19"></a>**[19]** Peitian Zhang et al. *Long Context Compression with Activation Beacon.* ICLR, 2025.

<a id="ref-20"></a>**[20]** DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.* [arXiv:2512.02556](https://arxiv.org/abs/2512.02556), 2025.

<a id="ref-21"></a>**[21]** DeepSeek-AI. *DeepSeek-V4 Technical Report.* [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), 2026.

<a id="ref-22"></a>**[22]** Yizhao Gao et al. *HySparse: Hybrid Sparse Attention with Oracle Token Selection and KV Cache Sharing.* [arXiv:2602.03560](https://arxiv.org/abs/2602.03560), 2026.

<a id="ref-23"></a>**[23]** Chen Zhang et al. *Efficient Context Scaling with LongCat ZigZag Attention.* [arXiv:2512.23966](https://arxiv.org/abs/2512.23966), 2025.

<a id="ref-24"></a>**[24]** Bowen Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models.* [arXiv:2309.00071](https://arxiv.org/abs/2309.00071), 2023.

<a id="ref-25"></a>**[25]** Yiran Ding et al. *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* [arXiv:2402.13753](https://arxiv.org/abs/2402.13753), 2024.

<a id="ref-26"></a>**[26]** Yao Fu et al. *Data Engineering for Scaling Language Models to 128K Context.* [arXiv:2402.10171](https://arxiv.org/abs/2402.10171), 2024.

<a id="ref-27"></a>**[27]** Tianyu Gao et al. *How to Train Long-Context Language Models (Effectively).* [arXiv:2410.02660](https://arxiv.org/abs/2410.02660), ACL 2025.

<a id="ref-28"></a>**[28]** Yijiong Yu et al. *Training With Paraphrasing the Original Text Teaches LLM to Better Retrieve in Long-Context Tasks.* [arXiv:2312.11193](https://arxiv.org/abs/2312.11193), 2023.

<a id="ref-29"></a>**[29]** Konwoo Kim et al. *Data-Efficient Pre-Training by Scaling Synthetic Megadocs.* [arXiv:2603.18534](https://arxiv.org/abs/2603.18534), 2026.

<a id="ref-30"></a>**[30]** Xin Cheng et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.* [arXiv:2601.07372](https://arxiv.org/abs/2601.07372), 2026.

<a id="ref-31"></a>**[31]** Alex L. Zhang, Tim Kraska, Omar Khattab. *Recursive Language Models.* [arXiv:2512.24601](https://arxiv.org/abs/2512.24601), 2025.

</div>
