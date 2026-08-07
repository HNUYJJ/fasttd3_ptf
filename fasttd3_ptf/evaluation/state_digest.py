"""Checkpoint 的**语义**状态摘要（P2.2 protocol §1）。

## 为什么不能用 raw file SHA

PyTorch 的 zip 序列化把**文件名 stem 写进 zip 内部 entry 根目录**：

    torch.save(obj, "foo_13000.pt")  → zip entry  foo_13000/data.pkl
    torch.save(obj, "foo_final.pt")  → zip entry  foo_final/data.pkl

同一个对象、不同文件名 ⇒ **SHA256 必然不同**（连等长文件名也不同）。
`train_ptf.py:880` 用 `_use_new_zipfile_serialization=True`，
`:3733` 在训练循环外、无新 learner update 的情况下写 `_final.pt`——
于是 `_100000.pt` 与 `_final.pt` 的字节必然不同，而状态可能完全一样。

P2.1 的 263/263 "AMBIGUOUS" 全部由此产生。**raw SHA 只是物理文件身份。**

## 计算规则（protocol §1.1 冻结）

**禁止重新 `torch.save` 后取 SHA**——那会再次引入文件名依赖，
正是本模块要修的问题本身。改为递归规范化编码，每项带类型标记，
故 ``1`` 与 ``"1"``、``[1,2]`` 与 ``(1,2)`` 不会碰撞。
"""

from __future__ import annotations

import hashlib

#: evaluator 实际消费的顶层键（protocol §11.2）。
#:
#: `p0_evaluator_v2.run_panel_v2` 经 `load_student` 只取 actor 与 obs_normalizer
#: （critic 与 critic_normalizer 在 eval 路径上**结构性不使用**）。
#: 若 evaluator 将来消费更多状态，**必须同步扩充这里**，
#: 否则 `evaluation_state_digest` 会低估差异。
EVALUATION_STATE_KEYS = ("actor_state_dict", "obs_normalizer_state")


def _encode(obj, out: bytearray, depth: int = 0) -> None:
    """把任意对象递归编码成规范化字节。类型标记前缀保证不同类型不碰撞。"""
    if depth > 64:                       # 防御性：异常深的结构不再展开
        out += b"X:depth"
        return

    if obj is None:
        out += b"N;"
        return
    if isinstance(obj, bool):            # bool 是 int 子类，必须先接住
        out += b"b:" + (b"1;" if obj else b"0;")
        return
    if isinstance(obj, int):
        out += b"i:" + repr(obj).encode() + b";"
        return
    if isinstance(obj, float):
        out += b"f:" + repr(obj).encode() + b";"   # repr 保往返精度
        return
    if isinstance(obj, str):
        raw = obj.encode("utf-8")
        out += b"S:" + str(len(raw)).encode() + b":" + raw + b";"
        return
    if isinstance(obj, (bytes, bytearray)):
        out += b"B:" + str(len(obj)).encode() + b":" + bytes(obj) + b";"
        return

    # torch.Tensor —— 鸭子类型识别，避免本模块硬依赖 torch
    if hasattr(obj, "detach") and hasattr(obj, "dtype") and hasattr(obj, "shape"):
        try:
            t = obj.detach().cpu()
            if hasattr(t, "is_contiguous") and not t.is_contiguous():
                t = t.contiguous()
            out += (b"T:" + str(t.dtype).encode() + b":"
                    + str(tuple(t.shape)).encode() + b":")
            out += t.numpy().tobytes()
            out += b";"
        except Exception as exc:  # noqa: BLE001
            out += b"T!err:" + repr(exc).encode() + b";"
        return

    if isinstance(obj, dict):
        out += b"D:" + str(len(obj)).encode() + b"{"
        # 按 key 的字符串形式排序：dict 顺序不应影响身份
        for k in sorted(obj, key=lambda x: (type(x).__name__, str(x))):
            _encode(k, out, depth + 1)
            out += b"="
            _encode(obj[k], out, depth + 1)
            out += b","
        out += b"};"
        return

    if isinstance(obj, (list, tuple)):
        tag = b"L" if isinstance(obj, list) else b"P"   # list / tuple 不混同
        out += tag + b":" + str(len(obj)).encode() + b"["
        for v in obj:
            _encode(v, out, depth + 1)
            out += b","
        out += b"];"
        return

    if isinstance(obj, (set, frozenset)):
        out += b"E:" + str(len(obj)).encode() + b"{"
        for v in sorted(obj, key=lambda x: (type(x).__name__, str(x))):
            _encode(v, out, depth + 1)
            out += b","
        out += b"};"
        return

    # numpy 标量 / 数组
    if hasattr(obj, "dtype") and hasattr(obj, "tobytes"):
        try:
            out += (b"A:" + str(obj.dtype).encode() + b":"
                    + str(getattr(obj, "shape", ())).encode() + b":"
                    + obj.tobytes() + b";")
        except Exception as exc:  # noqa: BLE001
            out += b"A!err:" + repr(exc).encode() + b";"
        return

    out += b"O:" + repr(obj).encode() + b";"     # 兜底：保证可编码不崩


def digest_object(obj) -> str:
    """任意对象的稳定递归摘要。"""
    buf = bytearray()
    _encode(obj, buf)
    return hashlib.sha256(bytes(buf)).hexdigest()


def evaluation_state_digest(state: dict) -> dict:
    """evaluator 实际消费的状态摘要（actor + obs normalizer）。

    返回 ``{"digest": ..., "keys_present": [...], "keys_missing": [...]}``。
    缺键**不静默**——缺 actor 的 checkpoint 不能与有 actor 的判为"评估等价"。
    """
    present = [k for k in EVALUATION_STATE_KEYS if k in (state or {})]
    missing = [k for k in EVALUATION_STATE_KEYS if k not in (state or {})]
    sub = {k: state[k] for k in present}
    return {
        "digest": digest_object(sub),
        "keys_present": present,
        "keys_missing": missing,
        "complete": not missing,
    }


def full_state_digest(state: dict) -> str:
    """checkpoint 全部逻辑内容的摘要（含 optimizer / qnet / 计数器等）。"""
    return digest_object(state)


def compare_states(a: dict, b: dict) -> dict:
    """比较两个已加载的 checkpoint state，返回三态判定所需的全部信息。

    ``verdict`` 取值见 P2.2 protocol §2：

    ``FINAL_LOGICAL_ALIAS``               eval 与 full 都相同
    ``EVAL_EQUIVALENT_STATE_DIVERGENCE``  eval 相同、full 不同（**不是失败**）
    ``FINAL_POLICY_DIVERGENCE``           eval 不同（**硬失败**）
    """
    ea, eb = evaluation_state_digest(a), evaluation_state_digest(b)
    fa, fb = full_state_digest(a), full_state_digest(b)

    if ea["digest"] != eb["digest"]:
        verdict = "FINAL_POLICY_DIVERGENCE"
    elif fa != fb:
        verdict = "EVAL_EQUIVALENT_STATE_DIVERGENCE"
    else:
        verdict = "FINAL_LOGICAL_ALIAS"

    differing = []
    if verdict == "EVAL_EQUIVALENT_STATE_DIVERGENCE":
        keys = sorted(set(a or {}) | set(b or {}))
        for k in keys:
            if k not in (a or {}) or k not in (b or {}):
                differing.append(k)
            elif digest_object(a[k]) != digest_object(b[k]):
                differing.append(k)

    return {
        "verdict": verdict,
        "eval_digest_a": ea["digest"], "eval_digest_b": eb["digest"],
        "eval_complete_a": ea["complete"], "eval_complete_b": eb["complete"],
        "eval_keys_missing_a": ea["keys_missing"], "eval_keys_missing_b": eb["keys_missing"],
        "full_digest_a": fa, "full_digest_b": fb,
        "differing_top_level_keys": differing,
    }
