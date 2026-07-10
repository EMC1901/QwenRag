"""测试 chunker 模块以及 Stage 6 的集成。"""

import pytest
from rag_preprocess.chunker import (
    estimate_tokens,
    split_long_block,
    drop_toc_heading_runs,
    format_article_range,
    extract_article_references,
    build_chunks,
    ChunkConfig,
    Chunk,
    StructuredDocument,
)
from rag_preprocess.law_structure import (
    StructuredBlock,
    build_section_path,
    LawLevel,
)


# ═══════════════════════════════════════════════════════════════
# estimate_tokens
# ═══════════════════════════════════════════════════════════════

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short():
    tokens = estimate_tokens("测试文本")
    assert tokens > 0


def test_estimate_tokens_longer():
    text = "这是一段较长的中文测试文本。" * 50
    tokens = estimate_tokens(text)
    assert tokens > 100


# ═══════════════════════════════════════════════════════════════
# split_long_block
# ═══════════════════════════════════════════════════════════════

def test_split_long_block_short():
    result = split_long_block("短文本", max_tokens=100)
    assert len(result) == 1
    assert result[0] == "短文本"


def test_split_long_block_long():
    text = "这是一段测试。" * 500
    result = split_long_block(text, max_tokens=100)
    assert len(result) > 1
    for part in result:
        assert len(part) > 0


def test_split_long_block_splits_at_period():
    """长文本应在句号处断开。"""
    text = "第一句话。" + "这是很长的内容。" * 100
    result = split_long_block(text, max_tokens=50)
    assert len(result) > 1
    for part in result:
        # 每段结尾应是句号（最后一段除外）
        assert part.endswith("。") or part == result[-1]


# ═══════════════════════════════════════════════════════════════
# build_chunks — 基本功能
# ═══════════════════════════════════════════════════════════════

def _make_structured_block(text, block_index=0, law_level=None,
                           section_path=None, article_no=None):
    return StructuredBlock(
        block_id=f"b{block_index}",
        text=text,
        law_level=law_level,
        section_path=section_path,
        article_no=article_no,
        block_index=block_index,
    )


def test_build_chunks_single_block():
    """单个 block → 单个 chunk。"""
    doc = StructuredDocument(
        doc_id="test_doc",
        title="测试法规",
        blocks=[_make_structured_block("第一条 测试内容。")],
    )
    chunks = build_chunks(doc, ChunkConfig(max_chunk_tokens=5000))
    assert len(chunks) == 1
    assert chunks[0].title == "测试法规"
    assert chunks[0].chunk_index == 0


def test_build_chunks_preserves_title():
    """chunk 应保留文档标题。"""
    doc = StructuredDocument(
        doc_id="test_doc",
        title="中华人民共和国测试法",
        blocks=[_make_structured_block("正文内容。")],
    )
    chunks = build_chunks(doc)
    assert len(chunks) == 1
    assert "中华人民共和国测试法" in chunks[0].chunk_text_for_embedding


def test_build_chunks_preserves_section_path():
    """chunk 应保留章节路径。"""
    doc = StructuredDocument(
        doc_id="test_doc",
        title="测试法规",
        blocks=[
            _make_structured_block("第一条 内容。",
                                   section_path="第一章 总则",
                                   article_no="第一条"),
        ],
    )
    chunks = build_chunks(doc)
    assert len(chunks) == 1
    assert chunks[0].section_path == "第一章 总则"
    assert chunks[0].article_no == "第一条"
    assert chunks[0].article_range == "第一条"


def test_format_article_range():
    assert format_article_range([]) is None
    assert format_article_range(["第一条"]) == "第一条"
    assert format_article_range(["第一条", "第一条"]) == "第一条"
    assert format_article_range(["第一条", "第二条", "第三条"]) == "第一条-第三条"


def test_extract_article_references_from_amendment_decision():
    text = (
        "一、将第四条修改为：“测试内容。”\n"
        "二、删去第八条。\n"
        "三、将第十四条改为第十三条。\n"
        "九、删去第二十四条至第二十七条。"
    )

    refs = extract_article_references(text)

    assert refs == [
        "第四条",
        "第八条",
        "第十四条",
        "第十三条",
        "第二十四条至第二十七条",
    ]


def test_build_chunks_sets_article_range_for_multiple_articles():
    """一个 chunk 合并多条时，应记录条文范围。"""
    doc = StructuredDocument(
        doc_id="test_doc",
        title="测试法规",
        blocks=[
            _make_structured_block(
                "第一条 内容。",
                block_index=0,
                law_level=LawLevel.ARTICLE,
                section_path="第一章 总则",
                article_no="第一条",
            ),
            _make_structured_block(
                "第二条 内容。",
                block_index=1,
                law_level=LawLevel.ARTICLE,
                section_path="第一章 总则",
                article_no="第二条",
            ),
            _make_structured_block(
                "第三条 内容。",
                block_index=2,
                law_level=LawLevel.ARTICLE,
                section_path="第一章 总则",
                article_no="第三条",
            ),
        ],
    )

    chunks = build_chunks(doc, ChunkConfig(target_chunk_tokens=5000, max_chunk_tokens=5000))

    assert len(chunks) == 1
    assert chunks[0].article_no == "第三条"
    assert chunks[0].article_range == "第一条-第三条"
    assert "条文范围：第一条-第三条" in chunks[0].chunk_text_for_embedding


def test_build_chunks_sets_article_range_from_references_when_no_article_no():
    """修改决定没有正式条号时，应从正文引用条号生成 article_range。"""
    doc = StructuredDocument(
        doc_id="test_amendment_doc",
        title="关于修改测试条例的决定",
        blocks=[
            _make_structured_block(
                "关于修改测试条例的决定",
                block_index=0,
            ),
            _make_structured_block(
                "一、将第四条修改为：“测试内容。”",
                block_index=1,
                law_level=LawLevel.ITEM,
            ),
            _make_structured_block(
                "二、删去第八条。",
                block_index=2,
                law_level=LawLevel.ITEM,
            ),
            _make_structured_block(
                "三、删去第二十四条至第二十七条。",
                block_index=3,
                law_level=LawLevel.ITEM,
            ),
        ],
    )

    chunks = build_chunks(doc, ChunkConfig(target_chunk_tokens=5000, max_chunk_tokens=5000))

    assert len(chunks) == 1
    assert chunks[0].article_no is None
    assert chunks[0].article_range == "第四条、第八条、第二十四条至第二十七条"
    assert "条文范围：第四条、第八条、第二十四条至第二十七条" in chunks[0].chunk_text_for_embedding


def test_build_chunks_splits_on_chapter():
    """遇到章标题时切分 chunk。"""
    doc = StructuredDocument(
        doc_id="test_doc",
        title="测试法规",
        blocks=[
            _make_structured_block("第一章 总则", law_level=LawLevel.CHAPTER,
                                   section_path="第一章 总则"),
            _make_structured_block("第一条 内容。",
                                   section_path="第一章 总则",
                                   article_no="第一条"),
            _make_structured_block("第二章 分则", law_level=LawLevel.CHAPTER,
                                   section_path="第二章 分则"),
            _make_structured_block("第十条 具体。",
                                   section_path="第二章 分则",
                                   article_no="第十条"),
        ],
    )
    chunks = build_chunks(doc, ChunkConfig(max_chunk_tokens=500, target_chunk_tokens=500))
    # 章标题触发切分
    assert len(chunks) >= 2


def test_build_chunks_does_not_leave_standalone_chapter_before_long_article():
    """章标题后接超长条文时，不应留下只有章标题的短 chunk。"""
    long_article = "第一条 " + "这是很长的条文内容，用于测试超长条文拆分。" * 80
    doc = StructuredDocument(
        doc_id="test_doc",
        title="测试法规",
        blocks=[
            _make_structured_block(
                "第一章 总则",
                block_index=0,
                law_level=LawLevel.CHAPTER,
                section_path="第一章 总则",
            ),
            _make_structured_block(
                long_article,
                block_index=1,
                law_level=LawLevel.ARTICLE,
                section_path="第一章 总则",
                article_no="第一条",
            ),
        ],
    )

    chunks = build_chunks(doc, ChunkConfig(max_chunk_tokens=80, target_chunk_tokens=80))

    assert len(chunks) > 1
    assert chunks[0].chunk_text.startswith("第一章 总则\n\n第一条")
    assert chunks[0].section_path == "第一章 总则"
    assert chunks[0].article_no == "第一条"
    assert chunks[0].token_count <= 80


def test_drop_toc_heading_runs_removes_repeated_chapter_headings():
    """连续多个同级章标题通常是目录，应过滤。"""
    blocks = [
        _make_structured_block("第一章 总则", 0, LawLevel.CHAPTER),
        _make_structured_block("第二章 管理", 1, LawLevel.CHAPTER),
        _make_structured_block("第三章 责任", 2, LawLevel.CHAPTER),
        _make_structured_block("第一条 正文。", 3, LawLevel.ARTICLE, article_no="第一条"),
    ]

    filtered = drop_toc_heading_runs(blocks)

    assert [b.text for b in filtered] == ["第一条 正文。"]


def test_drop_toc_heading_runs_keeps_hierarchical_headings_before_article():
    """逐级的编/章/节标题不是目录，应保留给正文做上下文。"""
    blocks = [
        _make_structured_block("第一编 总则", 0, LawLevel.PART),
        _make_structured_block("第一章 基本原则", 1, LawLevel.CHAPTER),
        _make_structured_block("第一节 适用范围", 2, LawLevel.SECTION),
        _make_structured_block("第一条 正文。", 3, LawLevel.ARTICLE, article_no="第一条"),
    ]

    filtered = drop_toc_heading_runs(blocks)

    assert [b.text for b in filtered] == [
        "第一编 总则",
        "第一章 基本原则",
        "第一节 适用范围",
        "第一条 正文。",
    ]


def test_build_chunks_context_header_format():
    """embedding 文本应包含法规标题。"""
    doc = StructuredDocument(
        doc_id="test_doc",
        title="中华人民共和国行政处罚法",
        blocks=[
            _make_structured_block(
                "第一条 为了规范行政处罚...",
                section_path="第一章 总则",
                article_no="第一条",
            ),
        ],
    )
    chunks = build_chunks(doc)
    assert len(chunks) == 1
    embedding_text = chunks[0].chunk_text_for_embedding
    assert "法规标题" in embedding_text
    assert "中华人民共和国行政处罚法" in embedding_text
    assert "章节路径" in embedding_text
    assert "第一章 总则" in embedding_text
    assert "条号" in embedding_text
    assert "第一条" in embedding_text
    assert "正文：" in embedding_text


# ═══════════════════════════════════════════════════════════════
# 集成测试：text_cleaner + law_structure + chunker
# ═══════════════════════════════════════════════════════════════

class TestStage6Integration:
    """验证 text_cleaner → law_structure → chunker 的完整流程。"""

    def test_full_pipeline(self):
        """完整流程：从解析后的 block 到最终 chunk。"""
        from rag_preprocess.text_cleaner import clean_text

        # 模拟 docx_parser 解析出的块
        raw_texts = [
            "第一章　总则",
            "第一条　为了规范行政处罚的设定和实施，",
            "保障和监督行政机关有效实施行政管理，",
            "根据宪法，制定本法。",
            "第二条　行政处罚的设定和实施，适用本法。",
            "— 1 —",  # 页码噪声
        ]

        # Step 1: 清洗
        cleaned = [clean_text(t) for t in raw_texts]
        # 噪声行被清洗为空
        non_empty = [t for t in cleaned if t]

        # Step 2: 构造 mock blocks
        class MockBlock:
            def __init__(self, text, idx):
                self.text = text
                self.block_id = f"block_{idx}"
                self.block_index = idx
                self.block_type = "paragraph"

        blocks = [MockBlock(t, i) for i, t in enumerate(non_empty)]

        # Step 3: 法规结构识别
        structured_blocks = build_section_path(blocks)

        # Step 4: 验证结构识别
        chapter_block = structured_blocks[0]
        assert chapter_block.law_level == LawLevel.CHAPTER
        assert "第一章" in chapter_block.text

        article1_block = structured_blocks[1]
        assert article1_block.law_level == LawLevel.ARTICLE
        assert article1_block.article_no == "第一条"

        # Step 5: 生成 chunks
        doc = StructuredDocument(
            doc_id="test_pipeline",
            title="测试法规",
            blocks=structured_blocks,
        )
        chunks = build_chunks(doc, ChunkConfig(max_chunk_tokens=500))

        # Step 6: 验证 chunk
        assert len(chunks) > 0
        for c in chunks:
            # 每个 chunk 应有文本
            assert len(c.chunk_text) > 0
            # chunk_id 非空
            assert c.chunk_id
            # token_count 合理
            assert c.token_count > 0

    def test_pipeline_no_structure(self):
        """无结构的文本也应正常产出 chunk。"""
        from rag_preprocess.text_cleaner import clean_text

        raw_texts = [
            "这是一段没有法规结构的普通文本。",
            "它包含多个段落。",
            "但没有任何章/节/条标记。",
        ]
        cleaned = [clean_text(t) for t in raw_texts]

        class MockBlock:
            def __init__(self, text, idx):
                self.text = text
                self.block_id = f"block_{idx}"
                self.block_index = idx
                self.block_type = "paragraph"

        blocks = [MockBlock(t, i) for i, t in enumerate(cleaned)]
        structured_blocks = build_section_path(blocks)

        # 所有 block 的 law_level 应为 None
        for sb in structured_blocks:
            assert sb.law_level is None

        doc = StructuredDocument(
            doc_id="test_no_structure",
            title="普通文档",
            blocks=structured_blocks,
        )
        chunks = build_chunks(doc)
        assert len(chunks) > 0
        # embedding 文本不含章节路径和条号
        assert "章节路径" not in chunks[0].chunk_text_for_embedding
        assert "条号" not in chunks[0].chunk_text_for_embedding

    def test_pipeline_chunk_text_not_empty(self):
        """所有 chunk 的文本不应为空。"""
        from rag_preprocess.text_cleaner import clean_text

        raw_texts = [
            "第一条 本法规定了测试程序。",
            "第二条 测试程序应当遵循以下原则：",
            "（一）公开原则；",
            "（二）公正原则；",
            "（三）效率原则。",
        ]
        cleaned = [clean_text(t) for t in raw_texts]

        class MockBlock:
            def __init__(self, text, idx):
                self.text = text
                self.block_id = f"block_{idx}"
                self.block_index = idx
                self.block_type = "paragraph"

        blocks = [MockBlock(t, i) for i, t in enumerate(cleaned)]
        structured_blocks = build_section_path(blocks)

        doc = StructuredDocument(
            doc_id="test_not_empty",
            title="测试程序规定",
            blocks=structured_blocks,
        )
        chunks = build_chunks(doc)

        for c in chunks:
            assert len(c.chunk_text.strip()) > 0, f"空 chunk: index={c.chunk_index}"
            assert len(c.chunk_text_for_embedding.strip()) > 0

    def test_structure_levels_recognized(self):
        """验证编/章/节/条/项全部可识别。"""
        texts = [
            "第一编 总则",
            "第一章 基本原则",
            "第一节 适用范围",
            "第一条 立法目的",
            "（一）具体事项",
            "一、子事项",
            "1. 细节",
        ]
        # 全部应被识别为非 unknown
        for t in texts:
            level = LawLevel.__dict__.get('_member_map_', {})
            from rag_preprocess.law_structure import detect_law_level
            result = detect_law_level(t)
            assert result is not None, f"'{t}' 未识别"
