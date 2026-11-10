# PRISM-V 数据契约 v1（路线 C：自动伪标 + 人工校验）

版本 1.0 · 2026-11-10 · 状态：待批准
实现位置：`prismv/data/`（schemas / contract / truth），测试 `tests/test_dataset_contract.py`

---

## 1. 目标与非目标

**目标**：让 PRISM-V 的三个任务（qa / localize / describe）都能被评分，并让验证层 GeoVer
的**判定质量**本身可测量——而不只是靠端到端准确率的变化间接推断。

**非目标**：不追求题目数量、不覆盖 MindCube 之外的新基准、不做训练数据生产。

核心判断：现有 MindCube 只提供 `gt_answer`（实测 `with_gt_entity=0`、`with_gt_relations=0`），
因此 localize 与 describe 不可评分，且"降低几何幻觉"缺少可测量代理。**要补的是真值层，不是题量。**

---

## 2. 范围冻结

| 级别 | 内容 |
|---|---|
| **P0（本次）** | 三层 Schema、契约校验器（R1–R10）、双真值推导器、分层抽样计划器、伪标精度估计、审计门（拒绝计算） |
| **P1（下一轮）** | 伪标生成器接真实 detector/segmenter、源数据集位姿对齐脚本、`verifier_gt_agreement` 指标入 `analysis` |
| **P2（明确不做）** | 自动质检替代人工、多基准扩展、题目改写/难度自动生成 |

---

## 3. 三层数据模型

```
scene_pack/<scene>.json   几何真值层，一场景一份，只写一次
items.jsonl               题目层，两层真值的 join key
label_audit.jsonl         人工核验台账，独立真值的唯一来源
entities_pred/*.json      模型预测实体，**永不与 scene_pack 合并**
```

### 字段 → 指标映射（无指标消费的字段即负债）

| 字段 | 层 | 支撑的指标 / 主张 | 缺失后果 |
|---|---|---|---|
| `frames[].convention` | 1 | 所有横向谓词 | 约定猜反 ⇒ 左右整体反向，且数据看起来正常 |
| `frames[].depth_conf` | 1 | 重建质量消融 | `conf_thres` 空转 |
| `objects[].points` + `point_conf` | 1 | `conf_thres` 消融 | 已实测：注入几何无置信度 ⇒ 该消融臂完全无差异 |
| `objects[].radius` | 1 | `occludes` | 无角直径 ⇒ 遮挡只能从质心判，必然错 |
| `objects[].aliases` | 1 | 选项→实体解析 | localize 分母缩水 |
| `gt_provenance.pose_source` | 1 | 循环性判定 | 一致率按构造≈1.0 |
| `gt_provenance.pipeline_hash` | 1 | 可复现性 | 标签器改版后数据不可复现 |
| `items[].ref_frame.view` | 2 | 验证器全部横向判定 | 从题干正则猜 ⇒ 解析失败即静默弃权 |
| `options[].geo_truth_indep` | 2 | `verifier_gt_agreement` | 主张 2 无证据 |
| `options[].geo_truth_recon` | 2 | 噪声地板（一致率） | 无法区分"验证器错"与"重建错" |
| `options[].margin_band` | 2 | 分层抽样 | 伪标精度被易例抬高 |
| `items[].geo_unique` | 2 | `constraint_set_decisive_rate` | 歧义题与唯一题混算一个数 |
| `items[].unscorable_reason` | 2 | 诚实分母 | 分母悄悄缩水 |
| `items[].gt_confidence` | 2 | 结果折扣 | 被手写 ⇒ R6 拒绝 |
| `items[].gt_relations` | 2 | describe 关系召回 | 只能算精确率，不可算召回 |
| `audit[].stratum` | 3 | 分层伪标精度 | 只能给被偏置的整体数 |
| `audit[].human_truths[]` | 3 | 伪标精度、κ | 人工侧本身无质量证据 |
| `items[].split` | 2 | 场景不相交切分 | 同场景进"留出集" |

---

## 4. 硬规则（校验器 R1–R10）

| 编号 | 规则 | 拦截的失效 | 失败级别 |
|---|---|---|---|
| R1 | Schema 合规 | 字段缺失 / 枚举越界 | violation |
| R2 | 切分场景不相交 | 同场景跨 split ⇒ 评测非留出 | violation |
| R3 | `pose_source=reconstruction` ⇒ 必须声明非独立 | **循环性**：用被测同款估计器造真值 | violation |
| R4 | 自动真值必须有 `pipeline_hash` | 标签器改版后不可复现 | violation |
| R5 | 有点无 `point_conf` | `conf_thres` 消融空转 | warning（显式 INERT） |
| R6 | 正确项无法接地必须声明 `unscorable_reason` | 分母悄悄缩水 | violation |
| R7 | `gt_confidence` 必须等于台账实测精度（±0.10） | 手写置信度 | violation |
| R8 | 层样本 < `min_n` ⇒ 精度报 `None` | 3 条样本的"精度"进表 | 设计约束 |
| R9 | `margin` 与 `margin_band` 自洽 | 分层抽样被偏置 | warning |
| R10 | describe 无 `gt_relations` | 精确率与召回混算 | warning |
| — | **审计门**：无独立真值且无核验子集 ⇒ 一致率**抛异常** | 报出≈1.0 的假一致率 | `ContractViolation` |

R3/R10 的"拒绝计算而非给个数"是刻意的：这类数字一旦生成就会被写进表格，警告拦不住它。

---

## 5. 人工校验协议

- **分层维度**：`relation × frame_kind × margin_band`，并给 `near_threshold` 设抽样下限。
  均匀随机抽样会偏向大 margin 易例，**系统性高估**伪标质量。
- 近阈值样本不足 ⇒ 计划器显式输出 `INSUFFICIENT_NEAR_THRESHOLD`（按关系去重），不静默少抽。
- **三态标注**（true / false / cannot tell），与验证器判定同构。二值标注会在最该检验的边界区
  伪造真值。
- 子集部分双人标注并报 κ；没有它，"人工真值"自身无质量证据。
- 伪标精度是**后面所有验证器结论的折扣因子**，必须进论文，按关系分别报并带 CI。

样本量：`min_n` 生产默认 30/层（测试用 5）。精确功效需按预期效应量算，尚未做（见扫雷 RED-2）。

---

## 6. 双真值与噪声地板

```
geo_truth_indep   传感器/COLMAP 位姿 + GT 物体几何  → 唯一可作证据的版本
geo_truth_recon   重建位姿 + 重建实体              → 验证器实际看到的
```

两版**共用 GeoVer 的谓词算术**，输入几何不同。这样它们的分歧只度量"几何错多少"，
而不是混入公式差异。该一致率即验证器的噪声地板：任何实现都不可能比它所读的几何
更接近独立真值。

不报这个数，读者会把每一处分歧都归因于验证器出错。

---

## 7. 验收标准（可执行）

| 编号 | 标准 | 验证方式 |
|---|---|---|
| A1 | 合规数据集 0 violation | `contract validate` |
| A2 | R1–R7 每条违规均被拦截 | `tests/test_dataset_contract.py`（20 项，全过） |
| A3 | 循环真值时一致率拒绝计算 | `test_gate_raises_on_circular_truth` |
| A4 | 参照系翻转 ⇒ 真值随之翻转 | `test_truth_is_three_state_and_view_relative` |
| A5 | 缺 convention 直接报错，不猜 | `test_truth_requires_explicit_convention` |
| A6 | 两种位姿约定归一后判定一致 | `test_cam_to_world_equates_to_world_to_cam` |
| A7 | 精度不足层报 `None` 而非 0.0 | `test_r7_underpowered_stratum_reports_none_not_zero` |
| A8 | 全仓库门禁与回归通过 | `pytest tests/ -q`：81 passed |

---

## 8. 扫雷审查

**RED（阻塞）**

1. **源位姿可得性未确认**——路线 C 成立的前提。`pose_source=reconstruction` 时审计门会持续
   拒绝，全量一致率永远算不出来。需确认 MindCube 场景能否对上 ARKitScenes / DL3DV 源场景。
2. **功效未计算**——各分层需要多少题才够，目前只有 `min_n=30` 的默认值，非计算结果。
   已实测证据：5 题时配对 CI 宽到 `[+0.2,+1.0]`，无结论力。
3. **detector/segmenter 未接**——本次交付的是接口骨架与契约，不含真实推理；无 GPU/权重。

**YELLOW**

4. `radius` 缺失时 `occludes` 只能从质心判——已给 warning，但未强制禁止出题。
5. 别名表覆盖决定 localize 分母，需按实际选项词表校核。
6. `pseudo_label_precision` 的 adjudicated 优先级（人工裁决 > 多数票 > cannot tell）需在协议文档中固定。

**GREEN**

7. JSON Schema 版本升级策略（当前 `const "1.0"` 硬锁）。
8. 数据集发布许可字段（已留 `license`，空值仅告警）。

---

## 9. 实现禁止事项（给下一轮实现者）

- 不得把 `entities_pred` 合并进 `scene_pack`——接地误差会变成真值误差，失败无法归因。
- 不得手写 `gt_confidence`；必须从 `label_audit` 推导（R7 会拒）。
- 不得为绕过审计门而把 `independent_of_verifier_input` 置真——那是把已知循环写成无循环。
- 不得为 `near_threshold` 不足而静默少抽；不足必须可见。
- 真值推导不得另写一套谓词公式；必须复用 `prismv.geover.predicates`（否则分歧不可解释）。
- 不得把"精确率"与"关系召回"合成一个 describe 分数。

---

## 10. 与现有代码的接缝

| 位置 | 改动 | 状态 |
|---|---|---|
| `prismv/data/{schemas,contract,truth}` | 新增 | 已交付 |
| `prismv/eval/datasets.py::task_coverage` | 扩成双真值覆盖矩阵 + 伪标精度列 | 待 P1 |
| `prismv/geover/grounding.py` | `load_entity_points` 应用 `point_conf`（消 R5） | 待 P1 |
| `prismv/analysis.py` | 新增 `verifier_gt_agreement`，经 `audit_gate` 调用 | 待 P1 |
| `prismv/eval/runner.py` | 任务记录透传 `ref_frame`，替代题干正则猜视角 | 待 P1 |

---

## 附：当前交付状态

- 契约测试 **20/20**，全仓库回归 **81 passed**，风格门禁 **7/7**
- CLI 可用：`validate` / `sample` / `precision`
- 未产生任何真实数据结论（无 GPU、无权重、无 API key）

---

内容由 AI 生成
