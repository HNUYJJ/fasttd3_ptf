"""HumanoidBench 任务分类学 v1：静态特征提取（F1–F6）。

严格遵循 docs/experiments/task_taxonomy_v1_prereg_20260729.md 冻结的 schema。
**不训练、不 rollout、不读取任何 U 标签。**

原则：
  · 全部特征机械地从 `humanoid_bench/envs/*.py`（AST 解析）与
    `humanoid_bench/assets/**/*.xml`（递归 include 解析）提取；
  · 无法机械确定的字段一律记 "unknown"，**不按直觉补值**；
  · 每个非显然特征附源码/XML 证据位置（文件:行）。

F1 用 AST 而非正则：需要提取的是分量的**定义指纹**（依赖的物理量表达式、
bounds、margin、sigmoid 类型），初版只比较分量名称导致把 Walk 与
ClimbingUpwards 误判为同构，此处不得重犯。
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

HB = Path(__file__).resolve().parents[2] / \
    "fasttd3_ptf/official_code/humanoid-bench/humanoid_bench"
ENVS, ASSETS = HB / "envs", HB / "assets"
UNKNOWN = "unknown"


# ---------------------------------------------------------------- 任务注册表
def task_registry() -> dict[str, str]:
    """task_name -> ClassName，直接从 env.py 的 TASKS 字面量解析。"""
    src = (HB / "env.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TASKS" for t in node.targets
        ):
            return {k.value: v.id for k, v in zip(node.value.keys, node.value.values)}
    raise RuntimeError("TASKS 表未找到")


def class_locations() -> dict[str, tuple[Path, ast.ClassDef, list[str]]]:
    """ClassName -> (文件, ClassDef 节点, 基类名列表)"""
    out = {}
    for f in sorted(ENVS.glob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                out[node.name] = (f, node, bases)
    return out


# ---------------------------------------------------------------- 常量解析
def module_constants(path: Path) -> dict[str, float]:
    """模块级数值常量（_STAND_HEIGHT / _WALK_SPEED / ...）。"""
    consts = {}
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                try:
                    consts[tgt.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return consts


def resolve_class_attr(cls_chain: list[ast.ClassDef], name: str):
    """沿 MRO 链找类属性的字面值（如 _move_speed），找不到返回 None。"""
    for cls in cls_chain:
        for node in cls.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        return node.value
    return None


# ---------------------------------------------------------------- F1 reward
class RewardExtractor(ast.NodeVisitor):
    """在 get_reward 函数体内收集 变量名 -> tolerance 调用指纹。"""

    def __init__(self, consts: dict, cls_consts: dict):
        self.defs: dict[str, dict] = {}
        self.consts, self.cls_consts = consts, cls_consts

    def _num(self, node):
        """把表达式求成数值；失败返回其源码文本或 UNKNOWN。"""
        try:
            return ast.literal_eval(node)
        except Exception:
            pass
        if isinstance(node, ast.Name):
            for tbl in (self.consts, self.cls_consts):
                if node.id in tbl:
                    return tbl[node.id]
            return node.id
        if isinstance(node, ast.Attribute):          # self._move_speed
            if node.attr in self.cls_consts:
                return self.cls_consts[node.attr]
            return f".{node.attr}"
        if isinstance(node, ast.BinOp):
            l, r = self._num(node.left), self._num(node.right)
            if isinstance(l, (int, float)) and isinstance(r, (int, float)):
                if isinstance(node.op, ast.Div):
                    return l / r
                if isinstance(node.op, ast.Mult):
                    return l * r
                if isinstance(node.op, ast.Add):
                    return l + r
                if isinstance(node.op, ast.Sub):
                    return l - r
        return UNKNOWN

    def _physical_quantity(self, node) -> str:
        """tolerance 的第一个参数：它依赖什么物理量。用源码文本作指纹。"""
        try:
            txt = ast.unparse(node)
        except Exception:
            return UNKNOWN
        return re.sub(r"\s+", "", txt)

    def _fingerprint(self, call: ast.Call) -> dict:
        fp = {"quantity": self._physical_quantity(call.args[0]) if call.args else UNKNOWN,
              "bounds": UNKNOWN, "margin": UNKNOWN, "sigmoid": "gaussian(default)",
              "value_at_margin": UNKNOWN}
        for kw in call.keywords:
            if kw.arg == "bounds":
                if isinstance(kw.value, ast.Tuple):
                    fp["bounds"] = [self._num(e) for e in kw.value.elts]
                else:
                    fp["bounds"] = self._num(kw.value)
            elif kw.arg == "margin":
                fp["margin"] = self._num(kw.value)
            elif kw.arg == "sigmoid":
                fp["sigmoid"] = self._num(kw.value)
            elif kw.arg == "value_at_margin":
                fp["value_at_margin"] = self._num(kw.value)
        return fp

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            calls = [n for n in ast.walk(node.value)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "tolerance"]
            if calls:
                fps = [self._fingerprint(c) for c in calls]
                try:
                    expr = re.sub(r"\s+", "", ast.unparse(node.value))
                except Exception:
                    expr = UNKNOWN
                self.defs[name] = {"n_tolerance": len(fps), "terms": fps,
                                   "expr_is_product_of_tolerances": len(fps) > 1,
                                   "expr": expr[:200]}
            else:
                try:
                    self.defs.setdefault(name, {"expr": re.sub(
                        r"\s+", "", ast.unparse(node.value))[:200], "n_tolerance": 0})
                except Exception:
                    pass
        self.generic_visit(node)


def parse_reward_composition(fn: ast.FunctionDef) -> dict:
    """解析 reward 的组合结构。

    必须处理四种此前漏掉的形式（均已查源码确认）：
      · maze     `reward = (加权和) * gate + bonus`   —— 顶层是 Add，非 Mult
      · cabinet  `reward = ...` 之后 `reward += 100 * subtask`  —— AugAssign
      · truck    `reward = 0` 之后多条 `reward += 1000/100/-100` —— 全靠 AugAssign
      · kitchen  `return bonus, ...` 且 `bonus = float(len(completions))` —— 返回变量非 reward
    """
    # 1) 定位返回的表达式或变量名
    ret_expr = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Return):
            ret_expr = node.value.elts[0] if isinstance(node.value, ast.Tuple) else node.value
            break
    if ret_expr is None:
        return {"kind": UNKNOWN, "text": UNKNOWN}

    var = ret_expr.id if isinstance(ret_expr, ast.Name) else None

    # 2) 收集该变量的全部 Assign 与 AugAssign；变量为 None 时直接用返回表达式
    exprs = []
    if var:
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == var for t in node.targets):
                exprs.append(("=", node.value))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == var:
                exprs.append((type(node.op).__name__, node.value))
    else:
        exprs.append(("=", ret_expr))
    if not exprs:
        exprs.append(("=", ret_expr))

    def unparse(e):
        try:
            return re.sub(r"\s+", "", ast.unparse(e))
        except Exception:
            return UNKNOWN

    texts = [f"{op} {unparse(e)}" for op, e in exprs]
    joined = " ".join(texts)

    # 3) 结构判据（按优先级，全部机械可判）
    ops, big_const, has_sub, has_count = set(), [], False, False
    gated = False
    for op, e in exprs:
        if op in ("Sub",):
            has_sub = True
        for n in ast.walk(e):
            if isinstance(n, ast.BinOp):
                ops.add(type(n.op).__name__)
                if isinstance(n.op, ast.Sub):
                    has_sub = True
                # (加性表达式) * 因子  →  门控
                if isinstance(n.op, ast.Mult):
                    for side in (n.left, n.right):
                        if isinstance(side, ast.BinOp) and isinstance(side.op, (ast.Add, ast.Sub)):
                            gated = True
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                    and abs(n.value) >= 100:
                big_const.append(n.value)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len":
                has_count = True

    if has_count:
        kind = "event_count"
    elif big_const:
        kind = "event_dominated"
    elif has_sub:
        kind = "penalty_unbounded"
    elif gated:
        kind = "gated"
    elif "Add" in ops:
        kind = "additive"
    elif "Mult" in ops:
        kind = "multiplicative"
    else:
        kind = UNKNOWN

    return {"kind": kind, "text": joined[:400], "uses_min": "min(" in joined,
            "ops": sorted(ops), "big_constants": sorted(set(big_const)),
            "n_assignments": len(exprs), "return_var": var or "<expr>"}


def extract_f1(cls_name: str, locs: dict) -> dict:
    """F1：reward 代数 + 分量定义指纹。沿 MRO 找最近的 get_reward。"""
    chain, seen = [], set()
    cur = cls_name
    while cur in locs and cur not in seen:
        seen.add(cur)
        f, node, bases = locs[cur]
        chain.append((f, node))
        cur = bases[0] if bases else None

    fn = owner_file = None
    for f, node in chain:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "get_reward":
                fn, owner_file, owner_node = item, f, node
                break
        if fn:
            break
    if fn is None:
        return {"reward_owner": UNKNOWN, "composition": {"kind": UNKNOWN}}

    consts = module_constants(owner_file)
    cls_consts = {}
    for name in ("_move_speed",):
        v = resolve_class_attr([n for _, n in chain], name)
        if v is not None:
            try:
                cls_consts[name] = ast.literal_eval(v)
            except Exception:
                if isinstance(v, ast.Name) and v.id in consts:
                    cls_consts[name] = consts[v.id]

    ex = RewardExtractor(consts, cls_consts)
    ex.visit(fn)
    return {
        "reward_owner": f"{owner_file.name}::{owner_node.name}",
        "reward_owner_line": owner_node.lineno,
        "composition": parse_reward_composition(fn),
        "component_fingerprints": {k: v for k, v in ex.defs.items()
                                   if v.get("n_tolerance", 0) > 0},
        "class_constants": cls_consts,
        "module_constants": {k: v for k, v in consts.items() if k.startswith("_")},
    }


# ---------------------------------------------------------------- F5 termination
def extract_f5(cls_name: str, locs: dict) -> dict:
    cur, seen = cls_name, set()
    while cur in locs and cur not in seen:
        seen.add(cur)
        f, node, bases = locs[cur]
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "get_terminated":
                try:
                    body = re.sub(r"\s+", " ", ast.unparse(item))[:240]
                except Exception:
                    body = UNKNOWN
                return {"owner": f"{f.name}::{node.name}", "line": item.lineno,
                        "text": body}
        cur = bases[0] if bases else None
    return {"owner": UNKNOWN, "text": UNKNOWN}


def main() -> None:
    reg, locs = task_registry(), class_locations()
    out = {}
    for task, cls in reg.items():
        out[task] = {"class": cls, "F1": extract_f1(cls, locs), "F5": extract_f5(cls, locs)}

    dst = Path(__file__).resolve().parents[2] / "docs/data/task_taxonomy_v1_f1f5.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"任务数 {len(out)}\n")
    print(f"{'task':18s} {'class':16s} {'reward owner':34s} {'kind':20s} min?")
    print("-" * 100)
    for t, v in out.items():
        c = v["F1"]["composition"]
        print(f"{t:18s} {v['class']:16s} {v['F1']['reward_owner']:34s} "
              f"{c['kind']:20s} {'Y' if c.get('uses_min') else ''}")
    print(f"\nsaved: {dst}")


if __name__ == "__main__":
    main()
