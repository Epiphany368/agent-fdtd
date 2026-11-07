# AgentFDTD (`agent-fdtd`) — v2.0.0

**语言 / Language:** [English](README.md) | [简体中文](README_zh.md)

**面向自主科研 Agent 的、由确定性物理契约与公证器核验的 FDTD 电磁仿真与光子逆向设计平台。**
AI Agent 负责“提案”（建模、发起仿真、优化与科学主张申报），确定性进程退出码与运行后公证器（Post-Flight Notary）负责“裁决”。

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Version: 2.0.0](https://img.shields.io/badge/version-2.0.0-blue.svg)](pyproject.toml)

---

## 为什么构建这个平台？

光子芯片与微纳光学设计高度依赖电磁仿真，但长期面临一个核心痛点：**跑出一个数字很容易，证明这个数字可信却很难。**

本平台将传统的“先跑仿真、后写一次性脚本检查”反转过来：**没有任何一项能力或结论可以在缺少可证伪判据的前提下宣称通过。** 平台中的每一项能力均绑定：
- **预先声明的验收判据与退出码契约**：由程序退出码（而非结果文件是否存在）决定验收结果；
- **强制负对照（Negative Control）**：专门构造必须触发失败的案例，确保每一道验证门禁真实有效；
- **可追溯运行账本（`runs/ledger.jsonl`）**：记录每次运行的输入快照哈希、求解器版本与退出状态；
- **证据档案（`docs/`）**：记录实测结果、复现命令与明确的适用边界。

---

## 快速开始（一键初始化 Agent 与开发环境）

```bash
git clone https://github.com/nillinjay/agent-fdtd
cd agent-fdtd
./tools/setup_agent.sh      # 自动创建 .venv，链接/初始化 TorchFDTD 求解器，安装 MCP 服务端并执行自检
```

如果你希望手动配置或链接外部的 `TorchFDTD` 目录：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 链接外部 TorchFDTD 检出目录：
tools/bootstrap_solver.sh /path/to/torchfdtd
```

执行基线自检与自动化测试套件（以退出码为准）：

```bash
./tools/run.sh configs/selftest-spine.json src/selftest_spine.py    # 期望返回 exit 0
.venv/bin/python -m unittest discover -s checks                     # 运行验证门禁与负对照测试集
.venv/bin/python -m unittest discover -s mcptests                   # 运行 MCP 服务端测试集
```

---

## 通过 AI Agent 驱动平台（MCP Server + `SKILL.md`）

仓库根目录自带 `SKILL.md`（Agent 技能清单）与 `.mcp.json`（零配置自动发现），支持 MCP 的智能体客户端（Claude Code、Antigravity、Cursor、OpenCode、Codex）克隆本仓库后即可直接调用平台的 **MCP 工具集**：

```bash
pip install -e .                        # 安装 agent-fdtd-mcp 与 torchfdtd-platform-mcp 入口
agent-fdtd-mcp                          # 本地 stdio 传输模式
agent-fdtd-mcp --http --port 8787       # 远程 Streamable HTTP 传输模式
```

### 核心仿真与验证工具

| 工具名称 | 已实现功能声明 |
|---|---|
| `create_project(...)` | 声明式创建 2D/3D 光学仿真场景（`configs/projects/<id>.json`），自动配置能量守恒监测面 |
| `validate_project(...)` | 多级运行前准入预检（时间步上限、总网格规模、显存准入预估与能量盒闭合性校验） |
| `submit_run(config, dry_run=True)` | `dry_run=True` 返回完整执行命令与预检结论；`dry_run=False` 加锁执行并返回 **`{ok, exit_status, run_id, kind}`** |
| `resume_run(run_id, dry_run=True)` | 从崩溃安全检查点日志实现逐位一致的断点续跑 |
| `inspect_breakpoint(run_id)` | 查询断点检查点进度、已完成/目标步数、槽位完整性及进程锁状态 |
| `get_certificate(run_id)` | 校验并返回运行后公证证书（`verification_certificate.json`） |
| `list_configs()` | 列出所有已登记的可执行配置及其用途 |
| `describe_run(config)` | 查询指定配置对应的驱动脚本、求解器角色与预声明判据 |
| `validate_config(config)` | 执行环境与配置结构合法性检查，返回 `{ok, errors[]}`（不启动仿真） |
| `read_ledger(limit)` | 读取 `runs/ledger.jsonl` 中最近 N 条真实运行记录 |
| `explain_exit(code)` | 解释平台退出码语义（区分正常通过、科学发现、预检拒绝与公证拦截） |

### v2.0 科研实验、结论裁决与版图交付工具（扩展实验面板）

| 工具名称 | 已实现功能声明 |
|---|---|
| `create_experiment(...)` | 创建跨求解器统一光子实验描述（支持对接 `TorchFDTD`、`RCWA`、`Meep`、`GDS` 后端并显式拦截不兼容配置） |
| `evaluate_claim(...)` | 在双级数值误差预算门禁下对比较型科学命题进行四态裁决（`SUPPORTED`、`REFUTED`、`INCONCLUSIVE`、`UNSUPPORTED`） |
| `export_gds(...)` | 导出光子器件 `.gds` 版图，执行 PDK 设计规则检查（DRC）与版图回读物理等价性验证 |
| `export_evidence_bundle(...)` | 将实验配置、谱系记录与公证证书打包导出为标准 RO-Crate 1.1 科研证据包 |

**核心设计原则：模型负责提案，平台负责裁决（The model proposes, the harness disposes）。** 没有任何一个 MCP 工具调用大模型，所有 `ok` 状态均严格由底层进程退出码决定。

---

## `v2.0.0` 已实现的核心功能声明

**平台已实现并通过实机硬件验收的功能模块（涵盖至 `LEG 57`）：**

- **零信任执行脊柱与运行后公证器（`LEG 43`, `LEG 54`, `LEG 56`）**：提供独占进程锁、源码快照、`manifest.json`、内核遥测与读出文件交叉核验及运行后公证检查（`checks/post_notary.py`，检测到数据篡改或契约违背时强制返回 `exit 66`）。
- **五层物理验证体系（`V1`–`V5`, `LEG 44`–`LEG 48`）**：
  - `V1` 解析解基准测试（真空传播、Fabry-Perot 标准具、色散介质反射、波导模式激励）；
  - `V2` 无源性（Passivity）与闭合包围盒能量守恒检验；
  - `V3` 多档网格加密收敛阶梯与渐近误差评估；
  - `V4` 跨独立开源求解器（**Meep** FDTD 与 **TORCWA** RCWA）自动对账验证；
  - `V5` 物理单位与功率归一化自动核查。
- **GPU 加速、多实例并批仿真与断点续跑（`LEG 45`, `LEG 46`, `LEG 51`, `LEG 52`）**：
  - GPU 正向与伴随加速执行，具备完整的吞吐率、图捕获与多级存储硬件基准画像能力；
  - 支持共享拓扑下的多结构正向并批评估、可微伴随反向并批求导与检查点显存预算自动管理；
  - 支持崩溃安全检查点与逐位一致（`0.0` 误差）中断续跑。
- **校准矢量模场重叠、工艺容差评估与可微逆向设计（`LEG 49`, `LEG 53`）**：
  - 具备最小特征尺寸约束的端到端可微光子逆向设计闭环，并对连续结构与离散二值化结构分别执行独立正向 FDTD 复核；
  - 提供经单位功率校准的矢量模场耦合效率计算与连续制造刻蚀偏置（$\pm \Delta w$）敏感度评估。
- **科学命题自动证伪门禁与多后端实验引擎（`LEG 54`, `LEG 55`）**：
  - 支持区分估计级（`estimated`）与认证级（`certified`）双级数值误差预算，自动裁决科学比较命题并生成消歧建议；
  - 提供求解器无关的实验中间表示、崩溃安全的实验谱系追踪数据库与 RO-Crate 1.1 标准证据包导出。
- **GDSII / PDK 版图交付与物理回读验证（`LEG 56`）**：
  - 支持标准 SOI 工艺层叠映射、DRC 几何规则校验、`.gds` 版图导出/回读，并对回读版图执行独立 FDTD 物理等价性核验。
- **独立隔离评分的科学证伪基准评测集（`LEG 57`）**：
  - 内置 24 个正负对照科研评测任务与物理隔离的独立评分器，支持错误验证率（FVR）统计与消融基准对比。

**平台诚实边界声明（不作过度宣称的范围）：**

- **截断纵向窗口下的渐变网格反射误差（`LEG 47 M4`, `partial`, `exit 57`）**：当纵向计算窗口截断衬底界面时，高压缩比渐变网格会引起反射率偏差超出严格阈值并返回 `exit 57`；高精度反射率评估需使用覆盖完整层叠的纵向窗口。
- **防篡改校验与硬件级密钥隔离的边界**：本地 SHA-256 与可选环境变量密钥签名可拦截指标伪造与单文件篡改；若需防御拥有运行目录完整写权限的本地特权进程重写全部文件，需配合外部独立控制面保管签名私钥。
- **定位说明**：本平台是面向开发者与 AI Agent 的代码优先（Code-First）电磁仿真与验证基础设施，不包含图形化 CAD 界面或托管云服务。

---

## 平台退出码字典（Exit Code Vocabulary）

| 退出码 | 物理与工程含义 |
|---|---|
| `0` | 当前运行携带的所有预声明判据与运行后公证检查全部通过 |
| `53` | 触发预声明的科学停止条件——属于合法的科学结论 |
| `54` | 所有已执行判据为真，但覆盖范围为 `partial` |
| `55` | 预声明的物理假设无法满足，停止运行而不做静默近似 |
| `56` | 对照基线未能复现预期输入，比较无效 |
| `57` | 实测物理量超出预声明阈值——属于受保护的**科学发现（Finding）** |
| `58` | 实际运行资源超出边界 |
| `59` | 显存不足（OOM），安全停止并报告，绝不静默篡改求解器参数 |
| `61` `62` `63` | 判据未通过 / 负对照缺失或失效 / 缺少判据定义块 |
| `64` `65` `69` | 参数用法错误 / 运行前预检拒绝 / Python 环境缺失（**未执行仿真**） |
| `66` | **运行后公证拦截（Post-Flight Notary Refusal）**：遥测不一致、物理守恒违背或结论/版图未通过公证核验 |

---

## 目录结构

| 路径 | 内容说明 |
|---|---|
| `src/` | 仿真驱动、逆向设计、实验引擎、命题门禁、版图交付与平台核心模块 |
| `checks/` | 验证门禁、运行前预检（`pre_*.py`）、运行后公证器（`post_notary.py`）及自动化测试套件 |
| `benchmarks/` | 独立隔离评分的科学证伪基准评测集 |
| `configs/` | 声明式运行配置文件 |
| `tools/` | 受控启动器（`run.sh` / `launch.py`）与环境引导工具 |
| `mcp_server/` | MCP 服务端（支持 stdio 与 Streamable HTTP） |
| `mcptests/` | MCP 服务端自动化测试套件 |
| `docs/` | 每个 Leg 的实测证据档案 |
| `runs/ledger.jsonl` | 追踪在版本库中的真实运行账本 |
| `PROVENANCE.md` | 外部基准、文献引用与求解器版本溯源记录 |

## 开源许可（License）

本项目采用 **MIT 许可证**（详见 [`LICENSE`](LICENSE)）。关联的 **TorchFDTD** 求解器同样采用 MIT 许可证，不包含任何商业闭源二进制文件或专有材料库依赖。
