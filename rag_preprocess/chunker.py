"""按法规结构切分 chunk。

优先按法规层级（编/章/节/条）切分，
在需要时才按 token 上限拆分长条文。
每个 chunk 保留标题、章节路径、条号等上下文信息。
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag_preprocess.law_structure import LawLevel


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class ChunkConfig:
    """chunk 配置。"""

    target_chunk_tokens: int = 700
    max_chunk_tokens: int = 1200
    overlap_tokens: int = 80
    # 字符级估算参数（中文字符 / token）
    chars_per_token: float = 0.7


@dataclass
class Chunk:
    """单个 chunk。"""

    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str                          # 纯正文
    chunk_text_for_embedding: str            # 带上下文头的文本（用于 embedding）
    title: str | None = None
    section_path: str | None = None          # 如 "第一章 总则 / 第一节 基本原则"
    article_no: str | None = None            # 如 "第十条"
    article_range: str | None = None         # 如 "第十条-第十二条"
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    token_count: int = 0
    vector_id: int | None = None
    embedding_status: str | None = None


@dataclass
class StructuredDocument:
    """结构化文档，用于 chunk 生成。"""

    doc_id: str
    title: str | None = None
    zdjg_name: str | None = None
    gbrq: str | None = None
    sxrq: str | None = None
    flxz: str | None = None
    sxx: str | None = None
    blocks: list = field(default_factory=list)  # StructuredBlock 列表


# ═══════════════════════════════════════════════════════════════
# Token 估算
# ═══════════════════════════════════════════════════════════════

def estimate_tokens(text: str, chars_per_token: float = 0.7) -> int:
    """估算 token 数。

    中文 1 字符大约 0.5-1 个 token，默认使用 0.7。
    对于纯中文文本，这个估算是偏保守的。
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


# ═══════════════════════════════════════════════════════════════
# chunk 生成
# ═══════════════════════════════════════════════════════════════

def build_chunks(
    doc: StructuredDocument,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """按法规结构生成 chunks。

    策略：
    1. 遍历 structured blocks
    2. 在编/章/节/条边界处优先切分
    3. 条内积累到接近 target_chunk_tokens 时切分
    4. 超过 max_chunk_tokens 时在句号处拆分

    Args:
        doc: 结构化文档（包含 StructuredBlock 列表）
        config: chunk 配置

    Returns:
        Chunk 列表
    """
    if config is None:
        config = ChunkConfig()

    blocks = drop_toc_heading_runs(doc.blocks)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    block_start = 0
    current_context_only = True

    # 收集当前 chunk 的上下文信息
    current_section_path: str | None = None
    current_article_no: str | None = None
    current_article_nos: list[str] = []

    for i, block in enumerate(blocks):
        # 提取文本和结构信息
        if hasattr(block, "text"):
            text = block.text
            section_path = getattr(block, "section_path", None)
            article_no = getattr(block, "article_no", None)
            law_level = getattr(block, "law_level", None)
        else:
            text = str(block)
            section_path = None
            article_no = None
            law_level = None

        if not text:
            continue

        block_tokens = estimate_tokens(text, config.chars_per_token)
        is_context_heading = law_level in (
            LawLevel.PART,
            LawLevel.CHAPTER,
            LawLevel.SECTION,
        )

        # 判断是否应该在此处切分
        should_split = False

        # 1. 遇到章/节标题 → 另起新 chunk
        if law_level is not None:
            if law_level in (LawLevel.PART, LawLevel.CHAPTER, LawLevel.SECTION):
                should_split = True

        # 2. 遇到条且当前 chunk 已有足够内容 → 另起新 chunk
        if (law_level == LawLevel.ARTICLE
                and current_tokens >= config.target_chunk_tokens * 0.5
                and len(current_parts) > 0):
            should_split = True

        # 3. 超过 max_chunk_tokens → 必须切分
        if (current_parts
                and not current_context_only
                and current_tokens + block_tokens > config.max_chunk_tokens):
            should_split = True

        # ── 执行切分 ──
        if should_split and current_parts:
            chunk = _make_chunk(
                doc=doc,
                parts=current_parts,
                config=config,
                chunk_index=len(chunks),
                block_start=block_start,
                block_end=i - 1,
                section_path=current_section_path,
                article_no=current_article_no,
                article_range=format_article_range(current_article_nos),
            )
            chunks.append(chunk)
            current_parts = []
            current_tokens = 0
            block_start = i
            current_context_only = True
            current_section_path = None
            current_article_no = None
            current_article_nos = []

        # ── 添加到当前 chunk ──
        current_parts.append(text)
        current_tokens += block_tokens
        if not is_context_heading:
            current_context_only = False

        # 更新上下文信息
        if section_path:
            current_section_path = section_path
        if article_no:
            current_article_no = article_no
            if not current_article_nos or current_article_nos[-1] != article_no:
                current_article_nos.append(article_no)

        # ── 处理超长 block ──
        while current_tokens > config.max_chunk_tokens and current_parts:
            # 最后一个 block 太长，需要拆分
            last_part = current_parts[-1]
            part_tokens = estimate_tokens(last_part, config.chars_per_token)

            if part_tokens > config.max_chunk_tokens:
                # 拆分长文本
                prefix_tokens = sum(
                    estimate_tokens(p, config.chars_per_token)
                    for p in current_parts[:-1]
                )
                first_part_max_tokens = max(
                    1,
                    config.max_chunk_tokens - prefix_tokens,
                )
                split_parts = split_long_block(
                    last_part,
                    first_part_max_tokens,
                    config.chars_per_token,
                )
                # 第一个拆分部分加入当前 chunk
                current_parts[-1] = split_parts[0]
                current_tokens = sum(
                    estimate_tokens(p, config.chars_per_token) for p in current_parts
                )
                # flush current chunk
                chunk = _make_chunk(
                    doc=doc,
                    parts=current_parts,
                    config=config,
                    chunk_index=len(chunks),
                    block_start=block_start,
                    block_end=i,
                    section_path=current_section_path,
                    article_no=current_article_no,
                    article_range=format_article_range(current_article_nos),
                )
                chunks.append(chunk)
                # 剩余部分作为新 chunk 的内容
                for sp in split_parts[1:]:
                    current_parts = [sp]
                    current_tokens = estimate_tokens(sp, config.chars_per_token)
                    block_start = i
                    if current_tokens <= config.max_chunk_tokens:
                        chunk = _make_chunk(
                            doc=doc,
                            parts=[sp],
                            config=config,
                            chunk_index=len(chunks),
                            block_start=i,
                            block_end=i,
                            section_path=current_section_path,
                            article_no=current_article_no,
                            article_range=format_article_range(
                                [current_article_no] if current_article_no else []
                            ),
                        )
                        chunks.append(chunk)
                        current_parts = []
                        current_tokens = 0
                        current_context_only = True
                if not current_parts:
                    block_start = i + 1
                    current_article_nos = []
                break
            else:
                break

    # ── 剩余内容 ──
    if current_parts:
        chunk = _make_chunk(
            doc=doc,
            parts=current_parts,
            config=config,
            chunk_index=len(chunks),
            block_start=block_start,
            block_end=len(blocks) - 1,
            section_path=current_section_path,
            article_no=current_article_no,
            article_range=format_article_range(current_article_nos),
        )
        chunks.append(chunk)

    return chunks


def drop_toc_heading_runs(blocks: list) -> list:
    """过滤目录式的连续结构标题。

    真实正文中常见 "第一编 / 第一章 / 第一节 / 第一条" 这种逐级进入正文的结构，
    应该保留；但有些文件开头会列出 "第一章 / 第二章 / 第三章" 这样的目录，
    这些标题没有正文内容，单独向量化价值很低，因此跳过。
    """
    context_levels = {LawLevel.PART, LawLevel.CHAPTER, LawLevel.SECTION}
    result = []
    pending = []
    seen_levels = set()
    repeated_same_level = False

    def flush_pending():
        nonlocal pending, seen_levels, repeated_same_level
        if pending and not repeated_same_level:
            result.extend(pending)
        pending = []
        seen_levels = set()
        repeated_same_level = False

    for block in blocks:
        law_level = getattr(block, "law_level", None)
        if law_level in context_levels:
            if law_level in seen_levels:
                repeated_same_level = True
            pending.append(block)
            seen_levels.add(law_level)
            continue

        flush_pending()
        result.append(block)

    flush_pending()
    return result


def format_article_range(article_nos: list[str]) -> str | None:
    """根据 chunk 内条号列表生成条文范围。"""
    unique_article_nos: list[str] = []
    seen: set[str] = set()
    for article_no in article_nos:
        if not article_no or article_no in seen:
            continue
        unique_article_nos.append(article_no)
        seen.add(article_no)

    if not unique_article_nos:
        return None
    if len(unique_article_nos) == 1:
        return unique_article_nos[0]
    return f"{unique_article_nos[0]}-{unique_article_nos[-1]}"


ARTICLE_REF_RE = re.compile(
    r"第[一二三四五六七八九十百千万零〇两]+条"
    r"(?:\s*[至到\-—－]\s*第[一二三四五六七八九十百千万零〇两]+条)?"
)


def extract_article_references(text: str) -> list[str]:
    """从非标准条文文本中提取被引用的条号或条号范围。

    例如修改决定中的 "将第四条修改为..."、"删去第二十四条至第二十七条"。
    该函数只用于 chunk 本身没有正式 article_no 时的兜底元数据。
    """
    refs: list[str] = []
    seen: set[str] = set()
    for match in ARTICLE_REF_RE.finditer(text):
        ref = re.sub(r"\s+", "", match.group(0))
        if ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
    return refs


def format_article_refs(refs: list[str]) -> str | None:
    """格式化正文中引用到的条号列表。"""
    if not refs:
        return None
    return "、".join(refs)


def _make_chunk(
    doc,
    parts: list[str],
    config: ChunkConfig,
    chunk_index: int,
    block_start: int,
    block_end: int,
    section_path: str | None = None,
    article_no: str | None = None,
    article_range: str | None = None,
) -> Chunk:
    """构建单个 Chunk 对象。"""
    body = "\n\n".join(parts)
    token_count = estimate_tokens(body, config.chars_per_token)
    effective_article_range = article_range
    if not effective_article_range:
        effective_article_range = format_article_refs(extract_article_references(body))

    # ── 构建 embedding 用文本（带上下文头） ──
    header_parts = []
    if doc.title:
        header_parts.append(f"法规标题：{doc.title}")
    if getattr(doc, "zdjg_name", None):
        header_parts.append(f"发布机关：{doc.zdjg_name}")
    if getattr(doc, "gbrq", None):
        header_parts.append(f"公布日期：{doc.gbrq}")
    if getattr(doc, "sxrq", None):
        header_parts.append(f"施行日期：{doc.sxrq}")
    if getattr(doc, "flxz", None):
        header_parts.append(f"法规性质：{doc.flxz}")
    if getattr(doc, "sxx", None):
        header_parts.append(f"时效性：{doc.sxx}")
    if section_path:
        header_parts.append(f"章节路径：{section_path}")
    if effective_article_range:
        header_parts.append(f"条文范围：{effective_article_range}")
    if article_no:
        header_parts.append(f"条号：{article_no}")

    if header_parts:
        header = "\n".join(header_parts)
        text_for_embedding = header + "\n\n正文：\n" + body
    else:
        text_for_embedding = body

    chunk_id = _generate_chunk_id(doc.doc_id, chunk_index, body)

    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        chunk_index=chunk_index,
        chunk_text=body,
        chunk_text_for_embedding=text_for_embedding,
        title=doc.title,
        section_path=section_path,
        article_no=article_no,
        article_range=effective_article_range,
        paragraph_start=block_start,
        paragraph_end=block_end,
        token_count=token_count,
    )


def _generate_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """生成稳定的 chunk_id。"""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    raw = f"{doc_id}:{chunk_index}:{text_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# 长文本拆分
# ═══════════════════════════════════════════════════════════════

def split_long_block(
    text: str,
    max_tokens: int,
    chars_per_token: float = 0.7,
) -> list[str]:
    """长条文拆分，优先在句号、分号、换行处断开。

    拆分原则：
    1. 优先在句号处断开
    2. 其次在分号处断开
    3. 再次在换行处断开
    4. 实在不行才按字符数强制截断

    Args:
        text: 待拆分的文本
        max_tokens: 每段最大 token 数
        chars_per_token: 字符/token 估算比例

    Returns:
        拆分后的文本段列表
    """
    max_chars = int(max_tokens * chars_per_token)

    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    remaining = text

    while len(remaining) > max_chars:
        # 取 max_chars 长度的前缀
        segment = remaining[:max_chars]

        # 寻找最佳切割点（按优先级）
        cut_point = -1

        # 1. 句号
        last_period = segment.rfind("。")
        if last_period > max_chars * 0.5:
            cut_point = last_period + 1

        # 2. 分号
        if cut_point < 0:
            last_semicolon = segment.rfind("；")
            if last_semicolon > max_chars * 0.5:
                cut_point = last_semicolon + 1

        # 3. 换行
        if cut_point < 0:
            last_newline = segment.rfind("\n")
            if last_newline > max_chars * 0.5:
                cut_point = last_newline + 1

        # 4. 逗号
        if cut_point < 0:
            last_comma = segment.rfind("，")
            if last_comma > max_chars * 0.5:
                cut_point = last_comma + 1

        # 5. 强制截断
        if cut_point < 0:
            cut_point = max_chars

        parts.append(remaining[:cut_point].strip())
        remaining = remaining[cut_point:].strip()

    if remaining:
        parts.append(remaining)

    return parts
