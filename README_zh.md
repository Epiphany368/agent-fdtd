# AgentFDTD (`agent-fdtd`) — v1.0.1

**语言 / Language:** [English](README.md) | [简体中文](README_zh.md)

**面向自主科研 Agent 的、由确定性物理契约与公证器核验的 FDTD 电磁仿真与光子逆向设计平台。**
AI Agent 拥有“提案权”（建模、发起仿真与优化），而确定性进程退出码与运行后公证器（Post-Flight Notary）拥有唯一的“裁决权”。

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Version: 1.0.1](https://img.shields.io/badge/version-1.0.1-blue.svg)](pyproject.toml)

---

## 为什么构建这个平台？

光子芯片与微纳光学设计高度依赖电磁仿真，但电磁仿真领域长期存在一个痛点：**跑出一个数字很容易，证明这个数字可信却极难。**
当一个透射率或耦合效率数字写进论文或流片评审报告时，六个月后往往没人能回答：*当时究竟校验了哪些物理守恒律？网格与时间窗真的收敛了吗？优化后的结构满足最小工艺特征尺寸吗？*

本项目源于真实的硅基光栅耦合器（Grating Coupler）研发需求，并将传统的“先跑仿真、后人工写一次性脚本检查”彻底倒置：**没有任何一项能力可以在缺少可证伪判据的前提下宣称完成。**

平台中的每一项能力（Leg）都强制绑定四要素：
- **预先声明的科学判据与退出码契约**：在运行前写入配置与规范，由程序退出码（而非文件是否存在）决定验收结果；
- **强制负对照（Negative Control）**：专门构造必须触发失败的案例——**一个从未触发过失败模式的门禁只是文字描述，不是真正的门禁**；
- **不可篡改的运行账本（`runs/ledger.jsonl`）**：记录每次运行的输入快照哈希、求解器版本、执行步数与退出状态；
- **长篇证据档案（`docs/`）**：完整记录实测数据、复现命令，以及**当前实验未能确定什么**。

---

## 快速开始（一键初始化 Agent 与开发环境）

```bash
git clone https://github.com/nillinjay/agent-fdtd
cd agent-fdtd
./tools/setup_agent.sh      # 自动创建 .venv，链接/初始化内置 TorchFDTD v1.1.7 求解器，安装 MCP 服务端并执行自检
```

如果你希望手动配置或链接外部的 `TorchFDTD` 源码目录：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 链接外部 TorchFDTD 检出目录（或直接使用内置 vendor/torchfdtd）：
tools/bootstrap_solver.sh /path/to/torchfdtd
```

执行一次基线自检与测试套件（读取退出码，而非仅看日志）：

```bash
./tools/run.sh configs/selftest-spine.json src/selftest_spine.py    # 期望返回 exit 0
.venv/bin/python -m unittest discover -s checks                     # 运行验证与负对照测试集
.venv/bin/python -m unittest discover -s mcptests                   # 运行 MCP 服务端测试集
```

---

## 通过 AI Agent 驱动平台（MCP Server + `SKILL.md`）

仓库根目录自带 `SKILL.md`（Agent 技能规范）与 `.mcp.json`（零配置自动挂载），任何支持 MCP 的智能体客户端（Claude Code、Antigravity、Cursor、OpenCode、Codex）克隆本仓库后即可直接调用 **11 个标准 MCP 工具**：

```bash
pip install -e .                        # 安装 agent-fdtd-mcp 与 torchfdtd-platform-mcp 入口
agent-fdtd-mcp                          # 本地 stdio 传输模式
agent-fdtd-mcp --http --port 8787       # 远程 Streamable HTTP 传输模式
```

| 工具名称 | 功能说明 |
|---|---|
| `create_project(...)` | 声明式创建 2D/3D 光学仿真场景（`configs/projects/<id>.json`），自动合成 4 面（2D）或 6 面（3D）Poynting 能量守恒包围盒 |
| `validate_project(...)` | 多级运行前准入预检（计算 CFL 最大时间步 `dt_fs`、总 Yee 网格数、显存准入预估与 Poynting 盒闭合性） |
| `submit_run(config, dry_run=True)` | `dry_run=True` 返回完整执行命令与预检结果；`dry_run=False` 正式加锁执行并返回 **`{ok, exit_status, run_id, kind}`** |
| `resume_run(run_id, dry_run=True)` | 从双槽位检查点日志（`ResidentRestartJournal`）实现逐位一致（`0.0` 误差）的断点续跑 |
| `inspect_breakpoint(run_id)` | 查询断点检查点状态、已完成/目标步数、双槽（`slot_a` / `slot_b`）SHA-256 完整性及进程锁状态 |
| `get_certificate(run_id)` | 校验并返回运行后公证证书（`verification_certificate.json`），交叉核对内核遥测与读出文件的 SHA-256 摘要 |
| `list_configs()` | 列出所有已登记的可执行配置及其用途 |
| `describe_run(config)` | 查看指定配置对应的驱动脚本、求解器角色与预声明验收判据 |
| `validate_config(config)` | 执行环境与配置合法性检查，返回 `{ok, errors[]}`（不启动仿真） |
| `read_ledger(limit)` | 原样读取 `runs/ledger.jsonl` 中最近 N 条真实运行记录 |
| `explain_exit(code)` | 解释平台退出码语义（严格区分“合法科学发现”、“预检拒绝”与“公证篡改拦截”） |

**核心设计铁律：模型负责提案，平台负责裁决（The model proposes, the harness disposes）。**
没有任何一个 MCP 工具会调用大模型，所有 `ok` 状态均严格派生自底层进程的退出码。

---

## `v1.0.1` 已验证的核心能力与诚实边界

### 已验证并锁入公证门禁的能力（经 RTX 5070 Ti `sm_120` 实测验证）：

- **零信任执行脊柱（`./tools/run.sh`, `LEG 43`）**：独占锁、源码快照、`manifest.json`、由求解器内核直接写盘的 `solver_telemetry.json`、运行后确定性公证器（`checks/post_notary.py`，执行遥测与读出文件双向 SHA-256 校验、证书载荷自哈希及可选 `TORCHFDTD_NOTARY_SECRET` HMAC-SHA256 签名，任何篡改或物理违背均强制返回 `exit 66`）。
- **五层物理验证体系（`V1`–`V5`）**：
  - `V1` 解析解基准库（真空传播、Fabry-Perot 平板、色散 Lorentz 反射、波导模式激励）；
  - `V2` 无源性（Passivity）与 2D 四面 / 3D 六面闭合 Poynting 包围盒能量守恒门禁（`box_closure_residual <= tau_box`）；
  - `V3` 单调网格加密收敛阶梯（`LEG 47 M1`：3D 窄齿波导三档网格 $T = [0.645227, 0.574380, 0.569926]$，Richardson 收敛阶 $p = 3.9916$）；
  - `V4` 跨求解器自动对账（`LEG 48`：TorchFDTD 对标开源 **Meep 1.34.0** 最大衍射级效率偏差 $9.04 \times 10^{-5} \le 0.015$，对标 **TORCWA RCWA** 严格耦合波分析偏差 $4.12 \times 10^{-3} \le 0.025$）；
  - `V5` 物理单位与归一化台账自动核查。
- **可微逆向设计闭环与制造形态学复核（`LEG 49` + `TorchFDTD v1.1.7`）**：
  - 三级 `M3` 工艺约束：圆锥空间滤波、平滑 $\tanh$ 投影延续、离散形态学开闭运算（强制最小齿宽/槽宽 $\ge 150\text{ nm}$，`0` 违规）及真实 Armijo 回溯线搜索；
  - **完整端到端 FDTD 伴随方向导数验证**：中心有限差分相对误差实测 $2.38 \times 10^{-8} \le 10^{-6}$（同时验证滤波-投影解析 Jacobian 相对误差 $1.07 \times 10^{-9} \le 10^{-6}$）；
  - **连续结构与制造后离散二值化结构双轨独立 FDTD 复核**：连续投影目标值从 `0.8537` 提升至 `0.8965`（`+0.0428`），制造投影后的离散二值化光栅（`b_realized`）经独立 FDTD 重算效率从 `0.7019` 提升至 `0.7335`（`+0.0317`）；
  - 时间可逆色散 Lorentz 介质精确伴随求导（`torchfdtd/reversible_lorentz.py`）与离散色散拟合（`fit_discrete_lorentz`）。
- **崩溃安全断点恢复与 GPU 加速（`LEG 45` & `LEG 46`）**：双槽乒乓检查点日志 `ResidentRestartJournal` 实现逐位一致（`0.0` 误差）中断续跑；分块融合注入与零分配观测环形缓冲在 RTX 5070 Ti 上实现 `1.70x–2.08x` 加速且保持 FP64 机器精度一致。

### `v1.0.1` 不作过度宣称的边界：

- **截断纵向窗口下的渐变网格模式注入误差（`LEG 47 M4`, `partial`, `exit 57`）**：在纵向截断的紧凑 N8 窗口（$y \in [-1.5, +1.5]\,\mu\text{m}$）上跑满 17,347 步时，因硅衬底界面位于窗口外，高压缩比渐变网格会引起 $|\Delta R|/R = 2.13\% > 0.1\%$ 并誠实返回 `exit 57`；扩展纵向窗口覆盖完整层叠（$y \in [-2.6, +2.25]\,\mu\text{m}$）后误差降至 $|\Delta T|/T = 1.64 \times 10^{-4} < 10^{-3}$。
- **防篡改哈希链 vs. 硬件级密钥隔离**：`checks/post_notary.py` 通过 SHA-256 哈希链与可选环境变量密钥 HMAC 拦截驱动脚本伪造指标或单文件篡改；若要防御拥有运行目录完整写权限的本地恶意进程同步重写全套文件，需配合外部控制面（如 Helix）持有独立签名私钥。

---

## 第二阶段演进路线图（`v1.1` $\to$ `v2.0` · Wave E–G: LEG 51–57）

下一阶段完整架构设计规范已定稿并收录于 [`docs/spec-agent-fdtd-v2.0.md`](docs/spec-agent-fdtd-v2.0.md)，围绕 **“核心可微求解器与 GPU 系统加速”** 与 **“计算科学自动证伪（Computational Scientific Falsification）”** 双核推进：

| 里程碑版本 | 阶段编号 | 核心交付内容与验收标志 |
|---|---|---|
| **`v1.1.0`** | **LEG 51** | **FDTD GPU 性能剖析与伴随内存画像** — 在 RTX 5070 Ti 上系统测量不同网格规模、步数（`1k–20k`）、`CudaStepGraphs (width=1..64)`、`run_tensor_batch (B=1..8)` 与多级检查点的真实耗时、有效显存带宽与重算倍率，冻结性能基线 |
| **`v1.1.0`** | **LEG 53** | **经单位功率校准的全矢量模式重叠与连续几何亚像素偏置** — 通过解析场 $\eta=1/0$ 及前后向分离自检的 2D/3D Lorentz 矢量模场重叠算子（$16 P_{\text{in}} P_{\text{fib}}$ 归一化），以及基于连续齿边界的 M1 亚像素占空比映射与 $\pm 10\text{ nm}$ 连续刻蚀偏置评估 |
| **`v1.1.0`** | **LEG 54** | **双级数值误差预算（Level A/B）与 `ScientificClaim` 自动证伪门禁** — 严格区分经验不确定度估计（`assurance_level = "estimated"`）与经独立求解器/数学证明包络的认证误差界（`"certified"`），对科学命题给出 `SUPPORTED / REFUTED / INCONCLUSIVE / UNSUPPORTED` 四态裁决并自动生成消歧实验处方 |
| **`v1.2.0`** | **LEG 52** | **`TensorPermittivityBatchAdapter`、可微并批 FDTD 与检查点成本优化** — 打通共享拓扑下多介电常数张量的正向并批与反向伴随指针表并批（`FusedBatchTransposeYeeCUDA`），以及基于实测带宽的检查点自动预算优化器 |
| **`v2.0.0`** | **LEG 55** | **求解器无关中间表示 `PhotonicExperimentIR` 与 SQLite WAL 权威实验引擎** — 支持编译至 `TorchFDTD / RCWA / Meep / GDS`（对不等价维度/边界显式返回 `UNSUPPORTED_BY_BACKEND`），建立基于 SQLite WAL 单一权威源与 Transactional Outbox 投影的实验关系谱系，支持导出 RO-Crate 1.1 证据包 |
| **`v2.0.0`** | **LEG 56** | **GDS/PDK 几何与物理双等价交付闭环** — 完整 SOI `LayerStack` 与 DRC 校验，并对导出的 `.gds` 版图执行多边形回读与独立 FDTD 物理等价验证 |
| **独立基准评测** | **LEG 57** | **独立隔离评分的 SciAgent 科学证伪基准集与消融实验** — 包含 24 个正负对照科研任务（评分器与被测 Agent 严格物理隔离），公布经验错误验证率（Empirical FVR 及 Clopper-Pearson 95% 置信上限）及相对于裸 `Python + TorchFDTD` 的同预算消融实验结果 |

---

## 平台退出码字典（Exit Code Vocabulary）

| 退出码 | 物理与工程含义 |
|---|---|
| `0` | 当前运行携带的所有预声明判据与运行后公证检查全部通过 |
| `53` | 触发预声明的科学停止条件（如网格加密阶梯出现非单调反转）——属于合法的科学结论 |
| `54` | 所有已执行判据为真，但覆盖范围为 `partial`（如大网格触及准入上限） |
| `55` | 预声明的物理假设无法满足，停止运行而不做静默近似 |
| `56` | 对照基线未能复现预期输入，比较无效 |
| `57` | 实测物理量超出预声明阈值——属于受保护的**科学发现（Finding）**，不得伪装为成功 |
| `58` | 实际运行资源超出边界 |
| `59` | 显存不足（OOM），安全停止并报告，绝不静默篡改求解器参数 |
| `64` / `65` / `69` | 参数用法错误 / 运行前预检拒绝 / Python 环境缺失（**未执行仿真**） |
| `66` | **运行后公证拦截（Post-Flight Notary Refusal）**：检测到遥测篡改、违反无源性、Poynting 盒未闭合、梯度失准或越权伪造科学结论 |

---

## 目录结构

| 路径 | 内容说明 |
|---|---|
| `src/` | 仿真驱动、逆向设计闭环、物理校验与平台核心模块 |
| `vendor/torchfdtd/` | 内置打包的 MIT 许可 `TorchFDTD v1.1.7` GPU 电磁求解器（在主工作台仓库中对应 `src/torchfdtd`） |
| `checks/` | 验证门禁、运行前预检（`pre_*.py`）、运行后公证器（`post_notary.py`）及全部正负对照测试集 |
| `configs/` | 每个 Leg 与项目的声明式配置文件 |
| `tools/` | 唯一受控启动器（`run.sh` / `launch.py`）、环境引导与公开脱敏导出工具 |
| `mcp_server/` | 11 工具 MCP 服务端（支持 stdio 与 Streamable HTTP） |
| `mcptests/` | MCP 服务端完整自动化测试套件 |
| `docs/` | 每个 Leg 的长篇科学证据档案与 `v2.0` 总体设计规范（`spec-agent-fdtd-v2.0.md`） |
| `output/` | 状态概览与架构演进图谱 |
| `runs/ledger.jsonl` | 追踪在版本库中的真实运行账本 |
| `PROVENANCE.md` | 外部基准、文献引用与求解器版本溯源记录 |

## 开源许可（License）

本项目采用 **MIT 许可证**（详见 [`LICENSE`](LICENSE)）。内置或关联的 **TorchFDTD** 求解器同样采用 MIT 许可证，不包含任何商业闭源二进制文件或专有材料库依赖。
