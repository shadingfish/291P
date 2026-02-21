# 分布式 LLM 训练与通信 — 背景知识（从零到工具）

本文档面向**未修读课程或尚未学完 L4–L7 的组员**，从动机到硬件、集合通信、多种并行方式与内存优化，串讲分布式大模型训练与通信开销的完整背景，并为「协作操作通信与内存开销计算工具」及**即将分配的任务（Switch/Mesh/Torus 拓扑、PP）**提供可直接查阅的知识基础。文中**专有名词均给出英文及简短解释**，便于写代码注释和 REFERENCES 时引用讲义出处。

---

## 阅读指南

### 如何使用本文档

- **第一次读**：按顺序读 **一 → 二 → 三 → 四 → 五**，建立「为什么分布式 → 硬件与拓扑 → 集合通信 → 四种并行」的完整图景。
- **做拓扑任务（Task 1/2/3）前**：重点读 **三（拓扑）、四（集合通信）**，以及 **名词表** 和 **公式与讲义出处**。
- **做 PP 任务（Task 4）前**：重点读 **1.3 训练过程、五.5.2 流水线并行、四.4.5 P2P**，以及名词表中的 P2P、Activation、Stage。
- **写公式注释时**：查 **公式与讲义出处** 表，在代码里注明对应 L4/L5/L6/L7 的 slide 或论文。

### 与讲义的对应

| 讲义 | 内容 | 本文章节 |
|------|------|----------|
| **L4** Lecture 4 LLM Training - Foundations | Ring/Tree AllReduce，AllGather，ReduceScatter，层次拓扑 | 三（拓扑）、四（集合通信） |
| **L5** Lecture 5 LLM Training - DP | ZeRO 内存、层次 Ring 公式、NCCL | 六（ZeRO）、四.4.4（层次 Ring） |
| **L6** Lecture 6 LLM Training - PP+TP | PP P2P、TP 每层 AllReduce | 五.5.2（PP）、五.5.3（TP）、四.4.5（P2P） |
| **L7** Lecture 7 LLM Training - CP | CP All2All、Ulysses | 五.5.4（CP）、四.4.1（All2All） |

### 与任务的关系（你需要懂什么）

| 任务 | 建议掌握的节 | 核心概念 |
|------|--------------|----------|
| **Task 1** SWITCH 拓扑 | 三、四、名词表 | Topology、Switch、B/α、五种 Collective、公式出处 |
| **Task 2** MESH 拓扑 | 三、四 | Mesh、2D 网格、n_x×n_y、沿行/列通信 |
| **Task 3** TORUS 拓扑 | 三、四 | Torus、环绕、与 Mesh 的步数/公式差异 |
| **Task 4** PP | 1.3、五.5.2、四.4.5、六（可选） | 训练流程、Activation、P2P、Stage、M_pp=2·batch·seq·hidden |
| **Task 5** 整合与演讲 | 全文 + 八（小结） | 工具各模块与概念的对应关系 |

---

## 名词表（中英 + 简短解释）

便于快速查阅和写注释时引用。按字母/类型分组。

| 英文 | 中文 | 简短解释 |
|------|------|----------|
| **Activation** | 激活 | 前向传播时每层的输出，需存下来供反向用；PP 的 stage 间传的就是激活（及梯度）。 |
| **AllReduce** | 全归约 | 每个 GPU 有一份数据，做完后**每个 GPU 都得到同一份归约结果**（如求和）。DP 梯度同步用。 |
| **AllGather** | 全收集 | 每人有一块，做完后**每人都有全部块拼成的完整数据**。ZeRO 里把分片参数 gather 回来。 |
| **All2All** | 全对全 | 每个 GPU 把自己的块发给**每一个**其他 GPU，每人收到来自所有人的一块。CP（如 Ulysses）用。 |
| **Alpha (α)** | 延迟 | 单次通信的固定延迟（秒），与数据量无关；常与带宽 B 一起刻画链路。 |
| **Bandwidth (B)** | 带宽 | 链路传输速率，单位 bytes/s（如 NVLink ~900 GB/s）。通信时间 ≈ M/B + α。 |
| **Broadcast** | 广播 | 一个 GPU 有完整数据，发给大家，每人得到一份完整拷贝。 |
| **Collective** | 集合通信 | 多 GPU 一起参与的通信操作（AllReduce、AllGather 等），区别于点对点 P2P。 |
| **CP** (Context Parallelism) | 上下文并行 | 按**序列长度**维切分，用 All2All 交换信息以完成长序列 attention。 |
| **DP** (Data Parallelism) | 数据并行 | 按 **batch** 维切数据，每卡一份完整模型，用 AllReduce 同步梯度。 |
| **FLOP / FLOPs** | 浮点运算（量） | FLOP = 一次浮点运算；FLOPs = 总运算量。训练量级常用 6×参数量×token 数估算。 |
| **Hierarchical** | 层次拓扑 | 先节点内（NVLink）、再节点间（InfiniBand）的两层结构。 |
| **InfiniBand** |  infiniband | 节点间高速网络，带宽约 50 GB/s/GPU，延迟高于 NVLink。 |
| **Mesh** | 网格 | 2D 网格，n_x×n_y=N，每 GPU 只与上下左右邻居相连。 |
| **M** | 张量大小（字节） | 通信中「张量大小 M」一般指该张量总字节数（如梯度 = 参数量×2）。 |
| **N** | GPU 数 | 参与一次集体通信的 GPU 数量（如 DP degree、TP degree）。 |
| **Node** | 节点 | 一台机器，内有多块 GPU，用 NVLink 互连。 |
| **NVLink** | nvlink | 节点内 GPU 互连，带宽约 900 GB/s，延迟低。 |
| **P2P** (Point-to-Point) | 点对点 | 两个 GPU 之间直接收发，不是集体通信。PP 的 stage 间用 P2P 传激活。 |
| **PP** (Pipeline Parallelism) | 流水线并行 | 按**层**切模型，层与层之间用 P2P 传激活和梯度。 |
| **ReduceScatter** | 归约并分片 | 先归约，再把结果按 GPU 切分，每人拿一块。ZeRO 梯度阶段用。 |
| **Ring** | 环 | GPU 排成环，每个只和左右邻居通信。AllReduce 可拆成 ReduceScatter+AllGather。 |
| **Switch** | 交换机（理想） | 理想化星型/全二分带宽：任意两 GPU 可视为经交换机直连，步数少。 |
| **Tensor** | 张量 | 多维数组；通信量常按「元素个数×每元素字节数」即张量字节数算。 |
| **TFLOPS** | 万亿次浮点/秒 | 10^12 FLOPs/s，常用来描述 GPU 算力。 |
| **Topology** | 拓扑 | GPU 之间谁和谁相连、链路 B/α 如何；决定集体通信的步数与延迟。 |
| **Torus** | 环面 | 2D 网格带环绕（首尾相连），步数/直径与 Mesh 不同。 |
| **TP** (Tensor Parallelism) | 张量并行 | 单层内权重按行/列切到多卡，用 AllReduce 等拼出完整结果。 |
| **Tree** | 树 | 树形结构，AllReduce 可「先沿树归约再广播」，步数 2⌈log N⌉。 |
| **ZeRO** (Zero Redundancy Optimizer) | 零冗余优化器 | 把优化器状态/梯度/参数分片到 N 张卡，降低每卡显存。 |

---

## 一、基础概念：FLOP、张量、训练过程与分布式

### 1.1 什么是 FLOP？（Floating Point Operation）

- **FLOP** = 一次浮点运算（一次加法、乘法等）。**FLOPs** 表示总运算量；**TFLOPS** = \(10^{12}\) FLOPs/s，即每秒一万亿次浮点运算，常用来描述 GPU 算力（如 H100 标称约 1979 TFLOPS FP8）。
- 训练总计算量常用近似：**总 FLOPs ≈ 6 × 参数量 × 训练 token 数**。推导：前向约 2·P·T（每参数每 token 约 2 FLOP），反向约 2 倍前向（见 L5：F+B(I)+B(W) 约 3 倍前向），合计约 **6·P·T**。

### 1.2 张量（Tensor）是什么？

- **张量**是多维数组的抽象：有 **shape** 和 **dtype**。通信里说的「张量大小 **M**」一般指**该张量占用的总字节数**（元素个数 × 每元素字节数）；例如 FP16 每元素 2 字节。
- **和工具的对应**：Collective 模块的输入「M」就是字节数；DP 的梯度 M = 参数量×2（FP16）；TP 每层 AllReduce 的 M 与 batch×seq×hidden 相关；PP 的 P2P 传激活，M_pp ≈ 2·batch·seq·hidden·2（前向+反向）。

**常见张量示例**（便于理解 M 的规模）：

| 是什么 | 形状示例 | 字节数（FP16） |
|--------|----------|----------------|
| 整模型梯度（DP AllReduce） | 与参数量同形状 | 70B×2 ≈ 140 GB |
| 一层线性权重/梯度 | [4096, 4096] | ≈ 33 MB |
| 一个 batch 激活（PP P2P） | [batch, seq, hidden] | batch×seq×hidden×2 |

### 1.3 训练的过程是什么？（Forward / Backward / Optimizer）

大模型训练可简化为「重复很多步」的流程：

1. **前向（Forward）**：输入一批 token，从第一层算到最后一层，得到 logits 或 loss。中间每层输出叫 **激活（Activation）**，要存下来供反向用。
2. **反向（Backward）**：从 loss 往回求导，得到每层参数的梯度。反向 FLOPs 约前向的 2 倍。
3. **优化器步（Optimizer step）**：用梯度更新参数（如 Adam）。**分布式时**，梯度需先在各卡间一致（DP 用 AllReduce），再更新。
4. 下一批数据，重复 1→2→3。

因此：**训练 = 前向 + 反向 + 梯度同步 + 参数更新**。  
**和任务的对应**：PP（Task 4）的通信发生在「层与层之间」传**激活**（前向）和**激活的梯度**（反向），即 P2P，不是 AllReduce；工具里要算的就是这段 P2P 的延迟与体积。

### 1.4 分布式是怎么做的？（切数据 / 切层 / 切维度 / 切序列）

- **切数据** → **DP**：每卡一份模型，不同卡吃不同 batch，用 **AllReduce** 同步梯度。
- **切层** → **PP**：不同卡算不同层，层间用 **P2P** 传激活和梯度。
- **切单层内权重** → **TP**：每卡只算矩阵的一块，用 **AllReduce/AllGather** 等拼出完整结果。
- **切序列** → **CP**：每卡处理一段序列，用 **All2All** 交换以完成 attention。

工具里「分析现有配置」就是：给定 DP/PP/TP/CP 度数和拓扑，分别算各维度的通信（AllReduce / P2P / All2All）与内存。

---

## 二、为什么要做分布式训练？

- **Scaling Laws**：模型越大、数据越多，效果越好，因此需要训练超大模型（如 70B、405B）。
- **单卡算不完**：例如 405B、15T tokens，总 FLOPs ≈ 6×405e9×15e12；单卡 H100 需数百年来量级。**必须把计算与通信分布到大量 GPU 上**，即分布式训练。

---

## 三、硬件与拓扑（Topology）：GPU 如何相连？

### 3.1 单卡 → 节点（Node）→ 集群

- **单卡**：一块 GPU（显存、算力）。
- **节点**：一台机器内多块 GPU，通常用 **NVLink** 互连，带宽高（约 **900 GB/s**）、延迟低。
- **节点间**：用 **InfiniBand** 等，每 GPU 约 **50 GB/s**，延迟更高。  
通信时间近似：**时间 ≈ 数据量/带宽 + 延迟**，即 \( M/B + \alpha \)。工具里用 **B**（bytes/s）和 **α**（秒）描述一条链路；层次拓扑下用 **B1/α1**（节点内）和 **B2/α2**（节点间）。  
**出处**：L4 slide 16（NVLink ~900 GB/s，IB ~50 GB/s）。

### 3.2 拓扑（Topology）是什么意思？

**拓扑**描述「GPU 之间谁和谁直接相连、链路 B/α 如何」。不同拓扑下，同一种集体通信的**步数**和**每步数据量**不同，所以延迟公式不同。

| 拓扑（英文） | 中文 | 直观理解 | 工具状态 |
|-------------|------|----------|----------|
| **Ring** | 环 | GPU 排成环，只和左右邻居通信 | 已实现 |
| **Tree** | 树 | 树形，AllReduce 可「先归约再广播」 | 已实现 |
| **Hierarchical** | 层次 | 先节点内 Ring、再节点间，对应 NVLink+IB | 已实现 |
| **Switch** | 交换机（理想） | 理想化星型/全二分：任意两 GPU 可视为直连，步数少（如 AllReduce 2 步） | **Task 1** |
| **Mesh** | 网格 | 2D 网格 n_x×n_y=N，只和上下左右邻居相连 | **Task 2** |
| **Torus** | 环面 | 2D 网格带环绕（首尾相连），步数/直径与 Mesh 不同 | **Task 3** |

工具里支持「灵活拓扑」就是为了：**给定拓扑类型和 N、B、α（或 B1/α1、B2/α2），能算出该拓扑下各集体操作的延迟与通信量**。实现新拓扑时要在注释中写清公式并注明出处（见 [docs/TASKS.md](TASKS.md)）。

---

## 四、集合通信（Collective）与 P2P

训练时经常需要「所有 GPU 一起做同一类操作」，这类操作叫 **集合通信（Collective）**。**P2P（Point-to-Point）** 则是两两之间的收发，不做「全体参与」的归约或广播。

### 4.1 常见集合通信类型（与并行维度的对应）

| 名称（英文） | 含义（直观） | 典型用途 | 工具中的 CollectiveKind |
|-------------|--------------|----------|--------------------------|
| **AllReduce** | 每人一份数据，做完每人都是同一份归约结果（如 sum） | DP 梯度同步 | ALL_REDUCE |
| **ReduceScatter** | 先归约，再按 GPU 切分，每人拿一块 | ZeRO 梯度阶段 | REDUCE_SCATTER |
| **AllGather** | 每人一块，做完每人都有全部块拼成的完整数据 | ZeRO 把分片参数 gather 回来 | ALL_GATHER |
| **Broadcast** | 一个 root 有完整数据，发给大家 | 发初始权重、输入 | BROADCAST |
| **All2All** | 每人把自己的块发给**每一个**其他 GPU | CP（如 Ulysses）沿序列维交换 | ALL_TO_ALL |

通信开销 = **延迟（步数 × 每步时间）+ 通信量（bytes）**。同一 collective 在不同**拓扑**下步数和每步数据量不同，故工具需要「拓扑 + 集体类型 + M」一起算。

### 4.2 AllReduce 的两种经典实现（公式与出处）

**Ring AllReduce**（适合大张量、打满带宽）：

- 拆成 **ReduceScatter + AllGather**，共 \(2(N-1)\) 步，每步传 \(M/N\)。
- **延迟**：\(2(N-1) \times (M/(NB) + \alpha)\)。  
- **出处**：L4 slide 34。工具中见 `_ring_allreduce_flat`，注释已标 L4 slide 34。

**Tree AllReduce**（步数少、适合小张量）：

- 先沿树 Reduce，再沿树 Broadcast，共 \(2\lceil\log_2 N\rceil\) 步，每步传整份 \(M\)。
- **延迟**：\(2\lceil\log_2 N\rceil \times (M/B + \alpha)\)。  
- **出处**：L4 slide 36。工具中见 `_tree_allreduce_flat`。

### 4.3 AllGather / ReduceScatter（Ring）

- 各 \((N-1)\) 步，每步 \(M/N\)，延迟 \((N-1)\times(M/(NB)+\alpha)\)。  
- **出处**：L4 slide 34。AllReduce = ReduceScatter + AllGather，故 Ring AllReduce 的 \(2(N-1)\) 步来自这两段。

### 4.4 层次拓扑下的 Ring（Hierarchical）

当 GPU 分属多节点时，「先节点内、再节点间」：

1. 节点内 ReduceScatter（用 B1、α1）；
2. 节点间对已缩成 1/N 的数据做一次归约（用 B2、α2）；
3. 节点内 AllGather（用 B1、α1）。

**出处**：L4 slide 38（层次思路），L5 slide 4（B1/α1、B2/α2 公式）。工具中见 `_hierarchical_ring_allreduce`、`_hierarchical_ring_allgather_reducescatter`，Case 2 时 intra_steps = gpus_per_node - 1。

### 4.5 P2P（Point-to-Point）与 PP

- **P2P**：两个 GPU 之间直接 Send/Recv，**不是**集体通信。**PP** 的 stage 与 stage 之间传**激活**（前向）和**激活的梯度**（反向），用的就是 P2P。
- **延迟模型**：单次 P2P 可近似为 **latency = M/B + α**（一次发送 M 字节，带宽 B，固定延迟 α）。若 stage 边界跨节点，用 B2/α2；节点内用 B1/α1。
- **出处**：L6（PP P2P）。Task 4 要在工具里实现 P2P 延迟函数，并在 analyze 中当 pp_degree>1 时计算 M_pp（≈ 2·batch·seq·hidden·2 bytes）并调用该函数，注释中写清公式与 L6。

### 4.6 公式与讲义出处汇总（写注释时查此表）

| 公式 / 概念 | 出处 | 在代码中的位置 |
|-------------|------|----------------|
| Ring AllReduce 延迟 \(2(N-1)(M/(NB)+\alpha)\) | L4 slide 34 | collective.py `_ring_allreduce_flat` |
| Tree AllReduce 延迟 \(2\lceil\log_2 N\rceil(M/B+\alpha)\) | L4 slide 36 | collective.py `_tree_allreduce_flat` |
| Ring AllGather / ReduceScatter \((N-1)(M/(NB)+\alpha)\) | L4 slide 34 | collective.py `_ring_allgather_or_reducescatter_flat` |
| 层次 Ring（B1/α1, B2/α2，gpus_per_node） | L4 slide 38, L5 slide 4 | collective.py `_hierarchical_ring_*` |
| 每 GPU 内存（无 ZeRO）2Ψ+2Ψ+12Ψ | L5 slide 18 | memory.py `per_gpu_memory_bytes` |
| ZeRO-1/2/3 显存公式 | L5 slides 21–26 | memory.py（扩展用） |
| PP P2P 通信 | L6 | Task 4：collective 或 p2p.py |
| CP All2All 体积 | L7 | collective.py ALL_TO_ALL，analyze 中 CP 用 |

---

## 五、并行维度：DP、PP、TP、CP

分布式训练的本质是：**把数据/模型/序列切到多卡，各算一块，再用 Collective 或 P2P 把结果合起来**。

### 5.1 数据并行（DP, Data Parallelism）

- **切什么**：按 **batch** 维切数据，每卡一份完整模型，处理不同样本。
- **通信**：每 step 结束后对梯度做 **AllReduce**（sum），使各卡梯度一致。
- **内存**：每卡存完整模型+梯度+优化器 → 显存大，故有 ZeRO（第六节）。
- **工具**：analyze 中 dp_degree>1 时，对「梯度张量 M = 参数量×2」调用 collective_latency_and_volume(ALL_REDUCE, M, topology)。**任意已实现的拓扑**（Ring/Tree/Hierarchical/Switch/Mesh/Torus）都会自动用于 DP。

### 5.2 流水线并行（PP, Pipeline Parallelism）

- **切什么**：按**层**切模型，GPU0 算前几层，GPU1 算接下来几层，依次类推；层与层之间是一个 **stage**。
- **通信**：stage 之间用 **P2P** 传 **激活**（前向）和**激活的梯度**（反向），**不是** AllReduce。
- **通信量**：与 **batch×seq×hidden** 成正比，每层边界传一次；近似 **M_pp = 2×batch×seq×hidden×2**（前向+反向，FP16）。
- **工具（Task 4）**：实现 P2P 延迟函数；在 analyze 中当 pp_degree>1 时计算 M_pp，调用 P2P，得到「每个 stage 边界的 P2P 延迟」并写入报告；CLI 增加 --pp。公式与出处需写在注释和 REFERENCES.md。

### 5.3 张量并行（TP, Tensor Parallelism）

- **切什么**：单层内**权重**按行/列切到多卡，每卡只算一部分矩阵乘，再用 AllReduce 等拼出完整结果。
- **通信**：每层通常 **2 次 AllReduce**，张量大小与 batch×seq×hidden（或 head 维）相关。
- **工具**：analyze 中 tp_degree>1 且给定了 seq_length、hidden_size 时，对「每层 AllReduce 的 M」调用 collective_latency_and_volume(ALL_REDUCE, M_tp, topology)。拓扑同上，自动支持。

### 5.4 上下文并行（CP, Context Parallelism）

- **切什么**：按**序列长度**维切分，每人算一段 Q/K/V，用 **All2All** 交换以看到全局上下文。
- **通信**：All2All，体积与 seq、hidden、CP 度相关（见 L7）。
- **工具**：collective 中已有 ALL_TO_ALL；analyze 中 cp_degree>1 时可用 M 调用 All2All 并加入报告（当前可选，非 Task 1–4 强制）。

### 5.5 四维并行与工具

一张 GPU 在逻辑上同时属于 DP rank、PP rank、TP rank、CP rank。工具「分析现有配置」即：给定 num_gpus、topology、dp/pp/tp/cp degree、模型与 batch/seq/hidden，分别算各维度的通信（DP/TP 用 AllReduce，PP 用 P2P，CP 用 All2All）和每卡内存。

---

## 六、ZeRO：在 DP 下省显存

DP 时每卡一份完整模型、梯度、优化器，显存随模型线性增长。**ZeRO（Zero Redundancy Optimizer）** 把这三类状态**分片**到 N 张卡上，降低每卡显存。

### 6.1 未用 ZeRO 时每卡要存什么

- **模型权重**：2Ψ（FP16）；**梯度**：2Ψ；**优化器状态**（AdamW）：12Ψ（FP32 一阶矩+二阶矩+master）。  
- **总显存**约 16Ψ（再乘 2 若算激活等）。  
- **出处**：L5 slide 18。工具中 memory 模块按此算 ZeRO-0；见 REFERENCES.md。

### 6.2 ZeRO 三阶段（简要）

- **ZeRO-1**：只分片**优化器状态**；ZeRO-2：再分片**梯度**；ZeRO-3：再分片**模型参数**，每卡 1/N，用前 AllGather 拼出再算。ZeRO-3 通信约 1.5× AllReduce。  
- **出处**：L5 slides 21–26。工具当前仅实现 ZeRO-0；扩展时在 memory.py 加 zero_stage 与对应公式，并在注释中标明出处。

---

## 七、通信与计算的重叠（Overlap）

- **DP**：Backward 中 AllReduce 可与其它层计算重叠；粗略要求通信时间 ≤ backward 时间（L5）。
- **TP**：AllGather/ReduceScatter 在层内关键路径上，可通过 fused kernel 与计算重叠。
- **PP**：P2P Send/Recv 可异步发出，减少空等。

工具输出通信时间与内存，便于判断瓶颈在算力还是带宽。

---

## 八、小结：与工具的对应关系（便于分工与汇报）

| 概念 | 在工具中的体现 | 相关任务 |
|------|----------------|----------|
| 拓扑（Ring/Tree/Hierarchical/Switch/Mesh/Torus） | **topology.py**：kind、N、B/α 或 B1/α1、B2/α2、gpus_per_node；**collective.py** 按拓扑选公式 | Task 1/2/3：实现新拓扑 + 公式 + 注释出处 |
| AllReduce / AllGather / ReduceScatter / Broadcast / All2All | **collective.py**：CollectiveKind、collective_latency_and_volume(M, topology) → latency、volume、steps | Task 1/2/3：为新拓扑实现各 collective 公式 |
| P2P | **collective.py** 或 **p2p.py**：P2P 延迟函数；**analyze.py** 中 pp_degree>1 时调用 | Task 4 |
| DP / TP / CP | **analyze.py**：根据 dp/tp/cp degree 建拓扑、调 collective（AllReduce 或 All2All） | 新拓扑实现后自动支持，无需每人单独做 |
| PP | **analyze.py**：pp_degree>1 时算 M_pp、调 P2P、写报告；**run_cli.py** --pp | Task 4 |
| ZeRO 阶段 | **memory.py**：ZeRO-0 已实现；ZeRO-1/2/3 为扩展 | 可选扩展 |
| 公式出处 | **REFERENCES.md** + 代码注释中标明 L4/L5/L6/L7 slide 或论文 | Task 1–4 均需在注释中写清公式与出处 |

把上述背景串起来即可理解：**为什么要分布式、硬件与拓扑如何、集合通信与 P2P 如何算、四种并行各用哪种通信、ZeRO 如何省显存**，以及这些在「协作操作通信与内存开销计算工具」里分别由哪个模块负责、**你负责的任务会动到哪些概念与文件**。写新公式时请务必在注释中写清公式并注明出处（见 [docs/TASKS.md](TASKS.md)）。
