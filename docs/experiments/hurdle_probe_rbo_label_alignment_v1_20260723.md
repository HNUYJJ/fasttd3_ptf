# Hurdle 既有 probe 与三 seed RBO 标签对齐

日期：2026-07-23  
冻结 RBO 标签：`run > walk > stand > scratch`（stand 仅 seed 1；
run/walk/scratch 为 3 seeds）

## 只读对齐结果

| 既有候选信号 | stand | walk | run | source 排序 | 是否对齐 RBO |
|---|---:|---:|---:|---|---|
| Transfer Map v1 full return | 74.46 | 93.69 | 157.14 | run > walk > stand | 是 |
| Transfer Map v1 `move` | 0.170 | 0.344 | 0.561 | run > walk > stand | 是 |
| v2 h=25 `reward_gain` | −0.224 | 2.947 | 2.413 | walk > run > stand | top 不对齐 |
| v2 h=25 `progress_gain` | −0.188 | 3.427 | 3.640 | run > walk > stand | 是 |
| v2 h=25 旧复合 `score` | −0.411 | 6.374 | 6.053 | walk > run > stand | top 不对齐 |
| v2 h=50 `reward_gain` | −0.434 | 7.755 | 11.932 | run > walk > stand | 是 |
| v2 h=50 `progress_gain` | −0.769 | 8.128 | 14.601 | run > walk > stand | 是 |

## 结论

1. source-specific target probe 不是完全无信息：Hurdle 的任务进度和较长
   reward-bearing coverage 都排出了真实 RBO 顺序。
2. 旧 h=25 复合 `score=reward_gain+progress_gain` 因 walk/run 很小的 reward
   差异把 top 排反，说明固定的未经校准特征求和不适合直接作 top-1 selector。
3. `safe_horizon=50` 对三个 source 完全相同，只能决定“可以执行多久”，不能承担
   “应该选择谁”。
4. full-episode return 在 Hurdle 对齐，但旧跨任务分析已证明它不能普遍预测
   transfer ROI；本轮不能把单任务对齐升级成通用迁移性指标。
5. 下一步必须加入 source-specific 负标签，检验这些候选能否拒绝有害 source，
   而不是继续在 Hurdle 上调权重。

## 负例选择

选择 `h1hand-crawl-v0`：

- 历史混合 RBO：safe≈656、rand≈700、scratch≈812，已知存在负迁移；
- Transfer Map v2 却给 walk/run h=25 reward gain `+6.35/+5.43`，h=50
  `+13.91/+13.66`；
- 这构成关键矛盾：如果单源 RBO 标签仍为负，则即时 reward-bearing probe 不能承担
  absolute admission；如果某一单源实际为正，则过去 crawl 负迁移可能来自混源稀释，
  需要把“有害 bank”和“有害 source”分开。

因此 crawl 单源等剂量 seed-1 screen 是下一项最小、可证伪且能改变指标设计的实验。

