"""清洗空行、页眉页脚、乱码、异常空格。

针对中文法规文档的专用清洗逻辑：
- 保留 "第X条"、"第一章"、"（一）" 等法规结构编号
- 去除页眉页脚、页码等噪声
- 合并因 docx 解析产生的异常断行
- 统一空白字符和标点符号
"""

import re
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# 噪声行检测
# ═══════════════════════════════════════════════════════════════

# 法规结构编号模式 —— 这些行绝对不能删除
PRESERVE_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千万零〇]+[章节条编]"),  # 第X章/节/条/编
    re.compile(r"^（[一二三四五六七八九十]+）"),                   # （一）（二）
    re.compile(r"^[一二三四五六七八九十]+[、，]"),                  # 一、 二、
    re.compile(r"^\d+[\.\、]"),                                   # 1. 2.
    re.compile(r"^第[一二三四五六七八九十百千万零〇]+款"),          # 第X款
    re.compile(r"^第[一二三四五六七八九十百千万零〇]+项"),          # 第X项
]

# 噪声行模式
NOISE_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),                              # 纯数字（页码）
    re.compile(r"^[\s\-_=*#~^]+$"),                               # 纯符号行
    re.compile(r"^- \d+ -$"),                                     # "- 1 -" 页码格式
    re.compile(r"^第\s*\d+\s*页$"),                               # "第 1 页"
    re.compile(r"^Page\s+\d+$", re.IGNORECASE),                   # "Page 1"
    re.compile(r"^[—\-—]\s*\d+\s*[—\-—]$"),                       # "— 1 —" 页码格式
    re.compile(r"^[—\-—]\s*\d+\s*\n?$"),                          # "— 1" 页码
    re.compile(r"^\d+/\d+$"),                                     # "1/100"
    re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日\s*$"),               # 独立日期行（可能是页脚）
    re.compile(r"^共\s*\d+\s*页$"),                               # "共 X 页"
    re.compile(r"^[\(（]\d+[\)）]\s*$"),                          # "(1)" 单独页码
]

# 常见页眉页脚关键词
HEADER_FOOTER_KEYWORDS = [
    "页眉", "页脚", "版权所有", "翻印必究", "内部资料",
    "机密", "绝密", "秘密", "内部文件",
]


def is_noise_line(text: str) -> bool:
    """判断是否是噪声行（页眉页脚、页码等）。

    注意：法规结构编号（第X条、第一章等）不会被误判为噪声。
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # 先检查是否是法规结构编号 —— 必须保留
    for pattern in PRESERVE_PATTERNS:
        if pattern.match(stripped):
            return False

    # 检查噪声模式
    for pattern in NOISE_PATTERNS:
        if pattern.match(stripped):
            return True

    # 检查页眉页脚关键词
    for kw in HEADER_FOOTER_KEYWORDS:
        if kw in stripped:
            return True

    # 短小且无中文字符的行可能是噪声
    if len(stripped) < 4 and not _contains_chinese(stripped):
        return True

    return False


def _contains_chinese(text: str) -> bool:
    """检查文本是否包含中文字符。"""
    return bool(re.search(r"[一-鿿]", text))


# ═══════════════════════════════════════════════════════════════
# 空格和空白字符规范化
# ═══════════════════════════════════════════════════════════════

def normalize_spaces(text: str) -> str:
    """统一空白字符。

    - 全角空格转半角
    - 多个空格合并为一个
    - \\r\\n / \\r 统一为 \\n
    - 制表符转空格
    """
    if not text:
        return ""
    # 全角空格 → 半角
    text = text.replace("　", " ")
    # \r\n → \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 制表符 → 空格
    text = text.replace("\t", " ")
    # 多个空格 → 一个
    text = re.sub(r" {2,}", " ", text)
    return text


# ═══════════════════════════════════════════════════════════════
# 异常断行合并
# ═══════════════════════════════════════════════════════════════

def merge_broken_lines(text: str) -> str:
    """合并在 docx 解析过程中产生的异常断行。

    规则：如果一行不以句号、分号、引号等结束标记结尾，
    且下一行不以"第"、数字编号等起始标记开头，则两行合并。

    这能修复类似：
        "本法所称行政"
        "许可，是指..."
    →
        "本法所称行政许可，是指..."
    """
    if not text:
        return ""

    lines = text.split("\n")
    if len(lines) <= 1:
        return text

    # 一行结尾不应合并的标点（说明语义完整）
    sentence_endings = {"。", "；", "：", "！", "？", "”", "」", "』", "）", ")", "》"}

    # 下一行开头不应合并的模式（说明是新段落/新条目开始）
    next_line_starters = [
        re.compile(r"^第[一二三四五六七八九十百千万零〇]+[章节条编款]"),
        re.compile(r"^（[一二三四五六七八九十]+）"),
        re.compile(r"^[一二三四五六七八九十]+[、，]"),
        re.compile(r"^\d+[\.\、]"),
        re.compile(r"^第[一二三四五六七八九十百千万零〇]+章"),
        re.compile(r"^第[一二三四五六七八九十百千万零〇]+节"),
        re.compile(r"^第[一二三四五六七八九十百千万零〇]+条"),
    ]

    merged: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            merged.append(line)
            continue

        if not merged:
            merged.append(line)
            continue

        prev = merged[-1].rstrip()

        # 前一行以句末标点结尾 → 不合并
        if prev and prev[-1] in sentence_endings:
            merged.append(line)
            continue

        # 当前行以起始标记开头 → 不合并（是新条目）
        is_new_entry = any(p.match(stripped) for p in next_line_starters)
        if is_new_entry:
            merged.append(line)
            continue

        # 前一行以逗号或顿号结尾 → 合并
        merge_endings = {"，", "、", "；", "：", "—", "“", "「", "『", "（", "("}
        if prev and prev[-1] in merge_endings:
            merged[-1] = prev + stripped
            continue

        # 前一行是纯中文且不以句号结尾 → 可能是断行，合并（加空格）
        if prev and _contains_chinese(prev) and _contains_chinese(stripped):
            # 判断是否可能是同一句被断开
            # 如果前一行的最后一个字符和当前行的第一个字符都不是标点 → 合并
            last_char = prev[-1]
            first_char = stripped[0]
            if (last_char not in sentence_endings
                    and last_char not in merge_endings
                    and first_char not in sentence_endings
                    and first_char not in {"第", "（", "(", " ", "　"}):
                merged[-1] = prev + stripped
                continue

        merged.append(line)

    return "\n".join(merged)


# ═══════════════════════════════════════════════════════════════
# 主清洗函数
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """清洗单段文本。

    执行步骤：
    1. 空白字符规范化
    2. 合并异常断行
    3. 去除噪声行
    4. 压缩多余空行
    5. 去除首尾空白
    """
    if not text:
        return ""

    # 1. 空白字符规范化
    text = normalize_spaces(text)

    # 2. 合并异常断行
    text = merge_broken_lines(text)

    # 3. 逐行去除噪声
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not is_noise_line(stripped):
            cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # 4. 压缩多余空行（3个以上空行 → 2个空行）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. 去除首尾空白
    text = text.strip()

    return text


def clean_blocks(
    blocks: list,
    clean_text_fn=clean_text,
    noise_fn=is_noise_line,
) -> list:
    """批量清洗 ParsedBlock 列表。

    每个 block.text 都会经过 clean_text 处理。
    噪声行会被标记但保留（由上层决定是否过滤）。
    法规则结构编号会被保留。

    Args:
        blocks: ParsedBlock 对象列表
        clean_text_fn: 文本清洗函数
        noise_fn: 噪声判断函数

    Returns:
        清洗后的 block 列表（原地修改并返回）
    """
    cleaned_count = 0
    noise_count = 0

    for b in blocks:
        original = b.text
        cleaned = clean_text_fn(original)

        if cleaned != original:
            cleaned_count += 1

        if not cleaned or noise_fn(cleaned):
            # 如果清洗后成为空文本或噪声，保留原始文本但标空
            # 这样不会丢失数据，上层可以过滤
            b.text = cleaned if cleaned else original
            noise_count += 1
        else:
            b.text = cleaned

    # 可选：如果后续需要返回统计信息
    # return blocks, {"cleaned": cleaned_count, "noise_lines": noise_count}

    return blocks
