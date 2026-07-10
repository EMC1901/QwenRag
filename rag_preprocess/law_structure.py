"""识别章、节、条、款、项等法规结构。

为法规文档的每个文本块识别其所属的法规层级，
并构建完整的章节路径（如 "第一章 总则 / 第一节 基本原则"）。

遵循中国法规的层级体系：
  编 > 章 > 节 > 条 > 款 > 项
"""

import re
from dataclasses import dataclass
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# 法规层级枚举
# ═══════════════════════════════════════════════════════════════

class LawLevel(Enum):
    """法规结构层级（从高到低）。"""

    PART = "编"        # 第X编 — 最高层级
    CHAPTER = "章"     # 第X章
    SECTION = "节"     # 第X节
    ARTICLE = "条"     # 第X条 — 最基本的规范单元
    PARAGRAPH = "款"   # 条内的自然段落
    ITEM = "项"        # （一）（二）或 1. 2.
    SUBITEM = "目"     # 项下的子项

    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════
# 正则模式
# ═══════════════════════════════════════════════════════════════

# 中文数字（支持到"百千万亿"）
CN_NUM = r"[一二三四五六七八九十百千万零〇]+"

# 层级识别正则
PART_RE = re.compile(rf"^第{CN_NUM}编")
CHAPTER_RE = re.compile(rf"^第{CN_NUM}章")
SECTION_RE = re.compile(rf"^第{CN_NUM}节")
ARTICLE_RE = re.compile(rf"^第{CN_NUM}条")
ARTICLE_LINE_RE = re.compile(rf"^\s*(第{CN_NUM}条)")
PARAGRAPH_RE = re.compile(rf"^第{CN_NUM}款")
ITEM_RE = re.compile(r"^（[一二三四五六七八九十]+）")           # （一）（二）
ITEM_CN_RE = re.compile(r"^[一二三四五六七八九十]+[、，]")        # 一、 二、
ITEM_NUM_RE = re.compile(r"^\d+[\.\、]")                       # 1. 2.
SUBITEM_RE = re.compile(r"^\d+\)")                              # 1) 2)

# 用于从文本中搜索条号（不限于行首）
ARTICLE_SEARCH_RE = re.compile(rf"第{CN_NUM}条")


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class StructuredBlock:
    """带有章节路径的结构化文本块。

    由 ParsedBlock 升级而来，附加了法规层级和路径信息。
    """

    block_id: str
    text: str
    law_level: LawLevel | None = None
    section_path: str | None = None   # 如 "第一章 总则 / 第一节 基本原则"
    article_no: str | None = None     # 如 "第十条"
    block_index: int = 0
    block_type: str = "paragraph"


@dataclass
class _SplitBlock:
    """一个 parsed block 被拆分后的内部块。"""

    text: str
    block_id: str
    block_index: int
    block_type: str = "paragraph"
    raw_text: str | None = None
    is_noise: bool = False


# ═══════════════════════════════════════════════════════════════
# 层级检测
# ═══════════════════════════════════════════════════════════════

def detect_law_level(text: str) -> LawLevel | None:
    """识别当前文本属于编、章、节、条、款、项。

    按优先级从高到低匹配，返回第一个匹配的层级。

    Args:
        text: 待识别的文本行

    Returns:
        匹配的 LawLevel，或 None（普通段落/款）
    """
    stripped = text.strip()

    if not stripped:
        return None

    # 按层级从高到低匹配
    if PART_RE.match(stripped):
        return LawLevel.PART
    if CHAPTER_RE.match(stripped):
        return LawLevel.CHAPTER
    if SECTION_RE.match(stripped):
        return LawLevel.SECTION
    if ARTICLE_RE.match(stripped):
        return LawLevel.ARTICLE
    if PARAGRAPH_RE.match(stripped):
        return LawLevel.PARAGRAPH
    if ITEM_RE.match(stripped):
        return LawLevel.ITEM
    if ITEM_CN_RE.match(stripped):
        return LawLevel.ITEM
    if ITEM_NUM_RE.match(stripped):
        return LawLevel.ITEM
    if SUBITEM_RE.match(stripped):
        return LawLevel.SUBITEM

    return None


def detect_law_level_for_block(block) -> LawLevel | None:
    """为 ParsedBlock 或类似对象检测法规层级。

    兼容 RagPreprocess 的 ParsedBlock 和 law_structure 的 StructuredBlock。
    """
    text = block.text if hasattr(block, "text") else str(block)
    return detect_law_level(text)


# ═══════════════════════════════════════════════════════════════
# 辅助判断函数
# ═══════════════════════════════════════════════════════════════

def is_chapter_title(text: str) -> bool:
    """是否为章标题。"""
    return bool(CHAPTER_RE.match(text.strip()))


def is_section_title(text: str) -> bool:
    """是否为节标题。"""
    return bool(SECTION_RE.match(text.strip()))


def is_article_start(text: str) -> bool:
    """是否为条的开始。"""
    return bool(ARTICLE_RE.match(text.strip()))


def is_item_start(text: str) -> bool:
    """是否为项的开始。"""
    return (bool(ITEM_RE.match(text.strip()))
            or bool(ITEM_CN_RE.match(text.strip()))
            or bool(ITEM_NUM_RE.match(text.strip())))


def is_part_title(text: str) -> bool:
    """是否为编标题。"""
    return bool(PART_RE.match(text.strip()))


# ═══════════════════════════════════════════════════════════════
# 条号提取
# ═══════════════════════════════════════════════════════════════

def extract_article_no(text: str) -> str | None:
    """从文本中提取条号，如 "第十条"、"第一百二十三条"。

    优先匹配行首，其次搜索文中。
    """
    stripped = text.strip()
    m = ARTICLE_RE.match(stripped)
    if m:
        return m.group()

    # 搜索文中出现的条号（用于正文引用）
    m = ARTICLE_SEARCH_RE.search(stripped)
    if m:
        return m.group()

    return None


def extract_article_start_no(text: str) -> str | None:
    """只在文本开头提取条号。

    `extract_article_no` 会搜索正文中的条号引用，例如“根据第十条规定”。
    阶段 6 的上下文状态机不能使用这种宽松搜索，否则普通段落可能被误判为
    当前条文开始。因此这里仅匹配行首的“第X条”。
    """
    m = ARTICLE_RE.match(text.strip())
    if m:
        return m.group()
    return None


def split_articles_in_block(block) -> list:
    """当一个 parsed block 内包含多条“第X条”时拆成多个块。

    有些 Word 文档会把一整章的多条法规放在一个段落/块里，例如：

        第十四条 ...
        第十五条 ...
        第十六条 ...

    如果不拆分，后续 chunk 会把十几条塞在一起。这里仅按“行首出现的
    第X条”拆分，避免把“根据第十条规定”这类正文引用误拆开。
    """
    text = block.text if hasattr(block, "text") else str(block)
    if not text:
        return [block]

    lines = text.splitlines()
    article_line_count = sum(1 for line in lines if ARTICLE_LINE_RE.match(line.strip()))
    if article_line_count <= 1:
        return [block]

    segments: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if ARTICLE_LINE_RE.match(stripped) and current:
            segments.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        segments.append(current)

    if len(segments) <= 1:
        return [block]

    original_block_id = getattr(block, "block_id", "")
    original_index = getattr(block, "block_index", 0)
    block_type = getattr(block, "block_type", "paragraph")
    raw_text = getattr(block, "raw_text", text)
    is_noise = getattr(block, "_is_noise", getattr(block, "is_noise", False))

    split_blocks: list[_SplitBlock] = []
    for part_index, segment_lines in enumerate(segments):
        segment_text = "\n".join(line.strip() for line in segment_lines).strip()
        if not segment_text:
            continue
        split_blocks.append(
            _SplitBlock(
                text=segment_text,
                block_id=f"{original_block_id}:article:{part_index}" if original_block_id else "",
                block_index=original_index * 1000 + part_index,
                block_type=block_type,
                raw_text=raw_text,
                is_noise=is_noise,
            )
        )

    return split_blocks or [block]


def extract_chapter_no(text: str) -> str | None:
    """从文本中提取章号，如 "第一章"。"""
    m = CHAPTER_RE.match(text.strip())
    if m:
        return m.group()
    return None


def extract_section_no(text: str) -> str | None:
    """从文本中提取节号，如 "第一节"。"""
    m = SECTION_RE.match(text.strip())
    if m:
        return m.group()
    return None


# ═══════════════════════════════════════════════════════════════
# 章节路径构建
# ═══════════════════════════════════════════════════════════════

def build_section_path(blocks: list) -> list[StructuredBlock]:
    """为每个文本块附加章节路径。

    遍历所有 block，维护一个章节层级上下文栈。
    遇到编/章/节标题时更新路径，每个 block 都附带当前路径。

    路径格式： "第X编 XXX / 第X章 XXX / 第X节 XXX"

    Args:
        blocks: 文本块列表，每个元素需要有 .text 属性（如 ParsedBlock）

    Returns:
        StructuredBlock 列表，带有 section_path 和 article_no
    """
    result: list[StructuredBlock] = []
    path_parts: list[str] = []
    current_article_no: str | None = None

    for i, b in enumerate(blocks):
        for split_b in split_articles_in_block(b):
            text = split_b.text if hasattr(split_b, "text") else str(split_b)
            stripped = text.strip()

            level = detect_law_level(stripped)
            article_start_no = extract_article_start_no(stripped)

            # ── 更新章节路径 ──
            # 进入新的编/章/节后，不能继续继承上一章最后一个条号。
            if level == LawLevel.PART:
                path_parts = [stripped]
                current_article_no = None

            elif level == LawLevel.CHAPTER:
                # 清除章/节，保留编，添加新章
                path_parts = [p for p in path_parts if "章" not in p and "节" not in p]
                path_parts.append(stripped)
                current_article_no = None

            elif level == LawLevel.SECTION:
                # 清除节，保留编/章，添加新节
                path_parts = [p for p in path_parts if "节" not in p]
                path_parts.append(stripped)
                current_article_no = None

            # ── 更新/继承条号 ──
            # 只有行首“第X条”才开启新条文；普通段落、款、项继承当前条号。
            if level == LawLevel.ARTICLE and article_start_no:
                current_article_no = article_start_no

            article_no = current_article_no

            # ── 构建 StructuredBlock ──
            block_id = getattr(split_b, "block_id", "")
            block_index = getattr(split_b, "block_index", i)
            block_type = getattr(split_b, "block_type", "paragraph")

            sb = StructuredBlock(
                block_id=block_id,
                text=text,
                law_level=level,
                section_path=" / ".join(path_parts) if path_parts else None,
                article_no=article_no,
                block_index=block_index,
                block_type=block_type,
            )
            sb.raw_text = getattr(split_b, "raw_text", text)
            sb.clean_text = text
            sb.is_noise = getattr(split_b, "is_noise", False)
            result.append(sb)

    return result


def detect_and_annotate_blocks(blocks: list) -> list[StructuredBlock]:
    """为 parsed_blocks 列表添加法规结构标注。

    这是 build_section_path 的别名，语义更明确。
    """
    return build_section_path(blocks)


# ═══════════════════════════════════════════════════════════════
# 序列化辅助
# ═══════════════════════════════════════════════════════════════

def structured_block_to_dict(sb: StructuredBlock) -> dict:
    """将 StructuredBlock 序列化为字典（用于 JSON 输出）。"""
    return {
        "block_id": sb.block_id,
        "text": sb.text,
        "law_level": sb.law_level.value if sb.law_level else None,
        "section_path": sb.section_path,
        "article_no": sb.article_no,
        "block_index": sb.block_index,
        "block_type": sb.block_type,
    }
