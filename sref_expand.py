#!/usr/bin/env python3
r"""sref_expand.py — 安全展开带引用的嵌套结构文本（纯标准库，单文件）。

输入格式（JSON 超集）：
  - 对象 {"k": v}、数组 [v, ...]、字符串/数字/true/false/null
  - 行注释：# 到行尾
  - 引用标记：$ref(路径)

路径语法：
  - 点号分隔对象键：a.b.c
  - 方括号表示数组下标：users[0]
  - 反斜杠转义分隔符：a\.b 表示键 "a.b"；weird\[key\] 表示键 "weird[key]"
  - 可选 $ 前缀表示文档根：$.a[0] 等价于 a[0]

共享规则（重要设计决策）：
  同一路径被多处引用时，每个引用点独立展开为一份【深拷贝】。
  理由：输出必须是一棵可 JSON 序列化、可校验的纯树；若共享同一子对象，
  输出层面无法表达别名（JSON 无 anchor/alias），反而会在后续处理中
  引入隐式共享导致的误修改。拷贝语义配合“活动路径栈”检测循环，
  配合深度上限防止指数膨胀。

截断规则：
  - 循环引用：引用目标已在当前展开栈中 -> 输出 {"$cycle": "<路径>"} 并告警
  - 深度超限：嵌套展开超过 --max-depth -> 输出 {"$truncated": "..."} 并告警
  - 路径不存在：输出 {"$unresolvedRef": "<路径>"}，报告引用所在行号，退出码 1

用法：
  python3 sref_expand.py 输入文件 [-o 输出.json] [--max-depth N]
  python3 sref_expand.py --selftest      # 运行内置自测样例
"""

import argparse
import json
import sys

DEFAULT_MAX_DEPTH = 64


class ParseError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.col = col

    def __str__(self):
        return f"第 {self.line} 行第 {self.col} 列: {self.msg}"


class Ref:
    """解析阶段的引用节点，保留原始文本与行号用于错误定位。"""
    __slots__ = ("raw", "segments", "line", "col")

    def __init__(self, raw, segments, line, col):
        self.raw = raw
        self.segments = segments  # list[str|int]
        self.line = line
        self.col = col


def parse_path(raw, line, col):
    """把路径文本解析为段列表：str=对象键，int=数组下标。支持反斜杠转义。"""
    text = raw.strip()
    if text.startswith("$"):
        text = text[1:]
        if text.startswith("."):
            text = text[1:]
    segments = []
    buf = []
    i = 0

    def flush():
        nonlocal buf
        if not buf:
            raise ParseError(f"路径 {raw!r} 中存在空段", line, col)
        segments.append("".join(buf))
        buf = []

    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 1
            if i >= len(text):
                raise ParseError(f"路径 {raw!r} 末尾出现孤立的反斜杠", line, col)
            buf.append(text[i])
            i += 1
        elif ch == ".":
            flush()
            i += 1
        elif ch == "[":
            if buf:
                flush()
            j = text.find("]", i)
            if j == -1:
                raise ParseError(f"路径 {raw!r} 中 '[' 未闭合", line, col)
            idx = text[i + 1:j].strip()
            if not idx.isdigit():
                raise ParseError(f"路径 {raw!r} 中非法下标 {idx!r}", line, col)
            segments.append(int(idx))
            i = j + 1
        elif ch == "]":
            raise ParseError(f"路径 {raw!r} 中出现多余的 ']'", line, col)
        else:
            buf.append(ch)
            i += 1
    if buf:
        flush()
    if not segments:
        raise ParseError("空路径", line, col)
    return segments


class Parser:
    def __init__(self, text):
        self.text = text
        self.i = 0
        self.line = 1
        self.col = 1

    def error(self, msg):
        raise ParseError(msg, self.line, self.col)

    def peek(self):
        return self.text[self.i] if self.i < len(self.text) else ""

    def advance(self):
        ch = self.text[self.i]
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip_ws(self):
        while True:
            while self.peek() and self.peek() in " \t\r\n":
                self.advance()
            if self.peek() == "#":
                while self.peek() and self.peek() != "\n":
                    self.advance()
            else:
                return

    def expect(self, ch):
        if self.peek() != ch:
            self.error(f"期望 {ch!r}，实际为 {self.peek()!r}")
        self.advance()

    def parse(self):
        value = self.parse_value()
        self.skip_ws()
        if self.peek():
            self.error("文档末尾存在多余内容")
        return value

    def parse_value(self):
        self.skip_ws()
        ch = self.peek()
        if ch == "{":
            return self.parse_object()
        if ch == "[":
            return self.parse_array()
        if ch == '"':
            return self.parse_string()
        if ch == "$":
            return self.parse_ref()
        if ch == "-" or ch.isdigit():
            return self.parse_number()
        for lit, val in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(lit, self.i):
                for _ in lit:
                    self.advance()
                return val
        self.error(f"无法识别的值，起始字符 {ch!r}")

    def parse_object(self):
        self.expect("{")
        obj = {}
        self.skip_ws()
        if self.peek() == "}":
            self.advance()
            return obj
        while True:
            self.skip_ws()
            if self.peek() != '"':
                self.error("对象键必须是双引号字符串")
            key = self.parse_string()
            self.skip_ws()
            self.expect(":")
            value = self.parse_value()
            if key in obj:
                self.error(f"重复的键 {key!r}")
            obj[key] = value
            self.skip_ws()
            if self.peek() == ",":
                self.advance()
                continue
            self.expect("}")
            return obj

    def parse_array(self):
        self.expect("[")
        arr = []
        self.skip_ws()
        if self.peek() == "]":
            self.advance()
            return arr
        while True:
            arr.append(self.parse_value())
            self.skip_ws()
            if self.peek() == ",":
                self.advance()
                continue
            self.expect("]")
            return arr

    def parse_string(self):
        self.expect('"')
        out = []
        escapes = {'"': '"', "\\": "\\", "/": "/",
                   "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
        while True:
            ch = self.peek()
            if ch == "":
                self.error("字符串未闭合")
            if ch == '"':
                self.advance()
                return "".join(out)
            if ch == "\\":
                self.advance()
                esc = self.advance()
                if esc == "u":
                    hexs = "".join(self.advance() for _ in range(4))
                    try:
                        out.append(chr(int(hexs, 16)))
                    except ValueError:
                        self.error(f"非法 unicode 转义 \\u{hexs}")
                elif esc in escapes:
                    out.append(escapes[esc])
                else:
                    self.error(f"非法转义 \\{esc}")
            else:
                out.append(self.advance())

    def parse_number(self):
        start = self.i
        if self.peek() == "-":
            self.advance()
        while self.peek().isdigit():
            self.advance()
        if self.peek() == ".":
            self.advance()
            while self.peek().isdigit():
                self.advance()
        if self.peek() in "eE":
            self.advance()
            if self.peek() in "+-":
                self.advance()
            while self.peek().isdigit():
                self.advance()
        text = self.text[start:self.i]
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                raise ParseError(f"非法数字 {text!r}", self.line, self.col)

    def parse_ref(self):
        line, col = self.line, self.col
        for _ in "$ref":
            self.advance()
        self.skip_ws()
        self.expect("(")
        raw = []
        while True:
            ch = self.peek()
            if ch == "":
                self.error("$ref 的路径未闭合（缺少 )）")
            if ch == "\\":
                raw.append(self.advance())
                if self.peek():
                    raw.append(self.advance())
                continue
            if ch == ")":
                self.advance()
                break
            raw.append(self.advance())
        raw_path = "".join(raw).strip()
        segments = parse_path(raw_path, line, col)
        return Ref(raw=raw_path, segments=segments, line=line, col=col)


def resolve(root, segments):
    """按段列表在解析树上定位节点；失败抛 KeyError(缺失的段)。"""
    node = root
    for seg in segments:
        if isinstance(seg, int):
            if not isinstance(node, list) or seg >= len(node):
                raise KeyError(f"[{seg}]")
            node = node[seg]
        else:
            if not isinstance(node, dict) or seg not in node:
                raise KeyError(seg)
            node = node[seg]
    return node


def expand(node, root, active, depth, max_depth, errors, warnings):
    """展开节点。active 是当前展开链上已解析的引用路径（用于循环检测）。"""
    if isinstance(node, Ref):
        try:
            target = resolve(root, node.segments)
        except KeyError as exc:
            errors.append(
                f"第 {node.line} 行: 引用 $ref({node.raw}) 无法解析："
                f"路径中缺少段 {exc.args[0]!r}"
            )
            return {"$unresolvedRef": node.raw}
        key = tuple(node.segments)
        if key in active:
            warnings.append(
                f"第 {node.line} 行: 检测到循环引用 $ref({node.raw})，已截断"
            )
            return {"$cycle": node.raw}
        return expand(target, root, active | {key}, depth + 1,
                      max_depth, errors, warnings)
    if isinstance(node, dict):
        if depth >= max_depth:
            warnings.append(f"展开深度超过上限 {max_depth}，对象已截断")
            return {"$truncated": f"depth limit {max_depth}"}
        return {k: expand(v, root, active, depth + 1, max_depth, errors, warnings)
                for k, v in node.items()}
    if isinstance(node, list):
        if depth >= max_depth:
            warnings.append(f"展开深度超过上限 {max_depth}，数组已截断")
            return [{"$truncated": f"depth limit {max_depth}"}]
        return [expand(v, root, active, depth + 1, max_depth, errors, warnings)
                for v in node]
    return node


def validate_tree(node):
    """校验展开结果：不得残留 Ref 节点，且必须可被 json 序列化。"""
    if isinstance(node, Ref):
        raise AssertionError("展开结果中残留未处理的 Ref 节点")
    if isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(k, str):
                raise AssertionError(f"非字符串键: {k!r}")
            validate_tree(v)
    elif isinstance(node, list):
        for v in node:
            validate_tree(v)
    json.dumps(node)  # 可序列化性检查


def run(text, max_depth):
    """解析 + 展开 + 校验。返回 (结果或None, errors, warnings)。"""
    try:
        tree = Parser(text).parse()
    except ParseError as exc:
        return None, [f"解析失败: {exc}"], []
    errors, warnings = [], []
    result = expand(tree, tree, frozenset(), 0, max_depth, errors, warnings)
    if not errors:
        validate_tree(result)
    return result, errors, warnings


SELFTESTS = [
    (
        "样例1：正常展开（共享引用 + 转义路径）",
        64,
        r'''
{
  "users": [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
  ],
  "first.user": $ref(users[0]),
  "weird[key]": {"x": 10},
  "copy_of_weird": $ref(weird\[key\]),
  "again_first_user": $ref($.users[0])   # 与 first.user 引用同一路径 -> 独立拷贝
}
''',
    ),
    (
        "样例2：引用路径不存在（错误定位）",
        64,
        r'''
{
  "a": {"b": 1},
  "c": $ref(a.b.c),
  "d": $ref(nope[2])
}
''',
    ),
    (
        "样例3：循环引用（截断）",
        64,
        r'''
{
  "node": {"value": 1, "next": $ref(node)}
}
''',
    ),
    (
        "样例4：深嵌套超过深度上限（截断，上限=8）",
        8,
        '{"a":' * 40 + "1" + "}" * 40,
    ),
]


def selftest():
    for title, max_depth, text in SELFTESTS:
        print("=" * 60)
        print(title)
        print("-" * 60)
        print("输入:")
        print(text.strip())
        result, errors, warnings = run(text, max_depth)
        print("-" * 60)
        if result is not None:
            print("展开结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        for w in warnings:
            print(f"[告警] {w}")
        for e in errors:
            print(f"[错误] {e}")
        print(f"退出状态: {'失败(1)' if errors else '成功(0)'}")
    print("=" * 60)
    print("自测完成")


def main(argv=None):
    ap = argparse.ArgumentParser(description="安全展开带引用的嵌套结构文本")
    ap.add_argument("input", nargs="?", help="输入文件（省略或 - 表示标准输入）")
    ap.add_argument("-o", "--output", help="输出文件（默认标准输出）")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                    help=f"展开深度上限（默认 {DEFAULT_MAX_DEPTH}）")
    ap.add_argument("--selftest", action="store_true", help="运行内置自测样例")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    if args.input and args.input != "-":
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    result, errors, warnings = run(text, args.max_depth)
    for w in warnings:
        print(f"[告警] {w}", file=sys.stderr)
    for e in errors:
        print(f"[错误] {e}", file=sys.stderr)
    if result is None:
        return 1
    out = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
