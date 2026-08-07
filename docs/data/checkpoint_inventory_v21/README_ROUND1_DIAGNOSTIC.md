# 首轮 sentinel（`sentinel_diagnostic_round1.json`）标记为 DIAGNOSTIC

2026-08-07。该文件是 P2.1 首轮 sentinel 的**原始运行输出**，原样保留。
它报 `SENTINEL_PASS`、退出码 0，但**覆盖不完整，不构成对预注册 §9 的验证**。

## 缺口

预注册 §9 第 3 项要求：

> 构造：同 family 同 step 不同 SHA、且无冻结证据可区分 → `AMBIGUOUS_EXECUTION`

**实现里漏了这一项**。`make_injected()` 只构造了三件（改错 seed / 改错 step /
不可解析文件名），没有构造"真正无法区分的执行歧义"。
故首轮输出中 `ambiguous_executions` 为空数组——
但那是**因为没有输入能触发它**，不是因为数据干净。

`AMBIGUOUS_EXECUTION` 的检测路径**一次都没有被执行**，判据真空成立。
这与 evaluator 那边 S3/S4 的 VACUOUS 是同一类问题：
前提不成立时条件式判据为真，而真空成立不是验证。

## 首轮仍然有效的观察（不因本声明作废）

以下是**正例**，首轮确实执行了对应路径：

```
P0 alias    h1hand-crawl-v0|p0_crawl_abstain|1|FORMAL@13000
            正式路径与 archive A 的 SHA 相同 → EXACT_ALIAS 去重
            **没有误报 AMBIGUOUS**，也没有 FORMAL_ALIAS_INTEGRITY_FAILURE
archive B   独立的 execution_instance，未与 A 冲突
注入件 3/3  按预期分类
racing      rck / rad 各命中 2 个（v2 的 SENTINEL_UNAVAILABLE 是关键词用错）
universe    独立发现 1587 个 .pt；vs v1：common=1552 / v1_only=109 / v2_only=35
```

## 处置

按 `CLAUDE.md §4.1`：本文件标 DIAGNOSTIC → hotfix commit 补第 3 项注入件 →
重新冻结 → 独立重跑。**请引用重跑后的 `sentinel.json`，不要引用本文件。**
