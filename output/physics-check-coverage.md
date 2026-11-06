# 物理检验覆盖率盘点 — torchfdtd-platform

**这份文件分两次读：第 11 条腿的原始盘点，和三条腿做完之后的现状。** 第 11 条腿只做盘点
——不改任何标签、不写新检查、不重跑仿真，只把「谁被检查过」数清楚。第 12、13、14 条腿随后
把那份缺口清单上的四组做掉了三组，第四组查出来是登记写错了。

| | 第 11 条腿盘点时（2026-10-18 上午） | 现在（三条腿之后） |
|---|---|---|
| 读了 | 33 次运行、2062 个数字、96 类字段 | **40 次运行、3234 个数字、106 类字段** |
| 物理量字段总数 | 25 | **25** |
| 其中有通过的检验 | 4 | **8** |
| 有检验但没生效（接线问题） | 2 | **0**（第 12 条腿修好了） |
| 没有任何检验，也没被决定关闭 | 13 | **1** |
| 不是缺口（决定已关闭问题） | 8 | **16** |
| 分类不出来的 | 0 | **0** |
| 仓库里已登记的检验 | 6 个 | **8 个**（多了 `V1-grating-zero-order` 与 `V1-slab-mode`） |
| 登记了但不存在 | `V4`（跨代码对比）—— 永远不能拿它当「已覆盖」 | 同左 |

**口径不变：钉住一个数字的读取平面、跨度、归一化，不等于从物理定律验证了这个数字。**
第 13 条腿钉住的六个字段只是从「没定义」变成「定义清楚、但仍然只与自己过去的结果一致」，
它们进的是 `settled`（决定已关闭问题），**不是** `covered`，**没有任何标签因此变成 `verified`**。

## 8 个有通过的检验的物理量

`T`、`R`（菲涅尔与艾里两个解析算例 + `V2-a`/`V2-b`）、`energy_residual`、
`empty_reflection_floor`、`zero_order_efficiency` 与其 `_a`/`_b`（第 14 条腿的光栅零级算例）、
`launch_neff`（第 14 条腿的平板 TM0 算例）。

## 唯一剩下的缺口

| 字段 | 为什么还没关 | 成本 |
|---|---|---|
| `zero_order_forward_power` | 它是求解器单位下的**绝对**功率，归一化口径要先由 `S4`（已做完）之外的一个决定定下来；没有闭式解能覆盖绝对功率 | CPU + 约 0.3 GPU 小时 |

## 第 11 条腿原来的 13 个缺口，现在各自的去处

| 组 | 字段 | 去处 |
|---|---|---|
| 1 | `J`、`P_up_reduced`、`P_in_reduced`、`unit_mode_flux_reduced`、`reference_power_reduced`、`eta_lat` | 第 13 条腿做完：口径钉住，进 `settled`。六个字段里包含全项目最大的单项不确定（过渡区 `5.6` 个百分点，已**声明为无界**） |
| 2 | `zero_order_efficiency` 与其 `_a`/`_b`、`zero_order_forward_power` | 第 14 条腿把前三个做完了：一个光栅零级算例同时覆盖它们。绝对功率仍开放，理由见上表 |
| 3 | `energy_spread`、`energy_spread_relative` | **登记前提本身错了。** 第 12 条腿查明它不是闭合盒的能量守恒量，而是已搁置的斜入射阵列扫描路线里「不同相位下送达能量的离散程度」。拿闭合盒去覆盖它只能靠名字，所以改为 `settled` 并写明原因 |
| 4 | `energy_residual`、`empty_reflection_floor` | 第 12 条腿修好了接线：不再靠字段名授予，而是当场用 `R`、`T` 重算、逐位相同才授予 |

## 25 个物理量字段的现状（机械输出，`src/leg11_coverage.py`）

| 字段 | 覆盖状态 |
|---|---|
| `R` | covered |
| `T` | covered |
| `energy_residual` | covered |
| `empty_reflection_floor` | covered |
| `zero_order_efficiency` | covered |
| `zero_order_efficiency_a` | covered |
| `zero_order_efficiency_b` | covered |
| `launch_neff` | covered |
| `J` | settled |
| `P_in_reduced` | settled |
| `P_up_reduced` | settled |
| `unit_mode_flux_reduced` | settled |
| `reference_power_reduced` | settled |
| `eta_lat` | settled |
| `energy_spread` | settled |
| `energy_spread_relative` | settled |
| `forward_total` | settled |
| `backward_total` | settled |
| `delivered_total` | settled |
| `drift_deg` | settled |
| `k_fixed_per_um` | settled |
| `launch_scores` | settled |
| `model_peak_eta_lat` | settled |
| `reconstruction_error` | settled |
| `zero_order_forward_power` | **open** |

## 两条读法上的提醒

* **`WIRED BUT NOT GRANTING` 现在是 0**：以前有两个字段「有检验但没生效」，那是接线问题而不是缺
  检查，第 12 条腿修好了，并逐份对比了 39 个运行的报告确认没有把没通过的东西授予 `verified`。
* **`UNCLASSIFIED` 现在是 0**，但它每天都要重新挣：这个数字要求每个被报出去的字段都能被归类。
  今天新增了三处规则（`error`、控制组自己的记账字段、以及把一个审计运行从物理量表里排除），
  否则它又会涨回去。

原始命令：`./.venv/bin/python src/leg11_coverage.py`（只读，不写文件、不跑仿真）。
