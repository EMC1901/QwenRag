"""测试 law_structure 模块。"""

import pytest
from rag_preprocess.law_structure import (
    LawLevel,
    detect_law_level,
    detect_law_level_for_block,
    extract_article_no,
    extract_chapter_no,
    extract_section_no,
    build_section_path,
    detect_and_annotate_blocks,
    split_articles_in_block,
    is_chapter_title,
    is_section_title,
    is_article_start,
    is_item_start,
    is_part_title,
    StructuredBlock,
    structured_block_to_dict,
    PART_RE,
    CHAPTER_RE,
    SECTION_RE,
    ARTICLE_RE,
    ITEM_RE,
    ITEM_CN_RE,
    ITEM_NUM_RE,
)


# ═══════════════════════════════════════════════════════════════
# 正则模式测试
# ═══════════════════════════════════════════════════════════════

class TestRegexPatterns:
    """验证正则模式能正确匹配法规结构编号。"""

    def test_part_re_matches(self):
        assert PART_RE.match("第一编 总则")
        assert PART_RE.match("第二编")
        assert PART_RE.match("第十编")

    def test_part_re_not_matches(self):
        assert not PART_RE.match("第一部分")  # "部分" 不是 "编"

    def test_chapter_re_matches(self):
        assert CHAPTER_RE.match("第一章 总则")
        assert CHAPTER_RE.match("第三章")
        assert CHAPTER_RE.match("第十二章 法律责任")

    def test_chapter_re_not_matches(self):
        assert not CHAPTER_RE.match("第一节")

    def test_section_re_matches(self):
        assert SECTION_RE.match("第一节 基本原则")
        assert SECTION_RE.match("第三节")

    def test_section_re_not_matches(self):
        assert not SECTION_RE.match("第一章")

    def test_article_re_matches(self):
        assert ARTICLE_RE.match("第一条 立法目的")
        assert ARTICLE_RE.match("第十条")
        assert ARTICLE_RE.match("第一百二十三条 施行日期")

    def test_article_re_not_matches(self):
        assert not ARTICLE_RE.match("第一章 总则")

    def test_item_re_bracket(self):
        assert ITEM_RE.match("（一）基本定义")
        assert ITEM_RE.match("（十）其他")

    def test_item_re_chinese_comma(self):
        assert ITEM_CN_RE.match("一、适用范围")
        assert ITEM_CN_RE.match("十、其他事项")

    def test_item_re_numeric(self):
        assert ITEM_NUM_RE.match("1. 定义")
        assert ITEM_NUM_RE.match("10、其他")

    def test_article_re_article_in_text(self):
        """条号可以在行中任意位置。"""
        from rag_preprocess.law_structure import ARTICLE_SEARCH_RE
        m = ARTICLE_SEARCH_RE.search("根据第十条 的规定")
        assert m is not None
        assert m.group() == "第十条"


# ═══════════════════════════════════════════════════════════════
# detect_law_level
# ═══════════════════════════════════════════════════════════════

class TestDetectLawLevel:
    def test_part(self):
        assert detect_law_level("第一编 总则") == LawLevel.PART

    def test_chapter(self):
        assert detect_law_level("第一章 总则") == LawLevel.CHAPTER

    def test_section(self):
        assert detect_law_level("第一节 基本原则") == LawLevel.SECTION

    def test_article(self):
        assert detect_law_level("第一条 立法目的") == LawLevel.ARTICLE

    def test_article_with_complex_number(self):
        assert detect_law_level("第一百二十三条 施行日期") == LawLevel.ARTICLE

    def test_item_bracket(self):
        assert detect_law_level("（一）基本定义") == LawLevel.ITEM

    def test_item_chinese_comma(self):
        assert detect_law_level("一、适用范围") == LawLevel.ITEM

    def test_item_numeric(self):
        assert detect_law_level("1. 定义") == LawLevel.ITEM

    def test_normal_paragraph(self):
        """普通段落文本 → None。"""
        assert detect_law_level("本法所称行政行为，是指...") is None

    def test_empty_text(self):
        assert detect_law_level("") is None

    def test_whitespace_text(self):
        assert detect_law_level("   ") is None

    def test_higher_level_priority(self):
        """同时匹配多个层级时，应返回最高层级。"""
        # "第一条" 不应该匹配章/节/编，只匹配条
        assert detect_law_level("第一条 测试") == LawLevel.ARTICLE


# ═══════════════════════════════════════════════════════════════
# 辅助判断函数
# ═══════════════════════════════════════════════════════════════

class TestHelperFunctions:
    def test_is_chapter_title(self):
        assert is_chapter_title("第一章 总则") is True
        assert is_chapter_title("第一条") is False

    def test_is_section_title(self):
        assert is_section_title("第一节 基本原则") is True
        assert is_section_title("第一章") is False

    def test_is_article_start(self):
        assert is_article_start("第一条 立法目的") is True
        assert is_article_start("第一章") is False

    def test_is_item_start_bracket(self):
        assert is_item_start("（一）定义") is True

    def test_is_item_start_chinese(self):
        assert is_item_start("一、适用范围") is True

    def test_is_item_start_numeric(self):
        assert is_item_start("1. 定义") is True

    def test_is_part_title(self):
        assert is_part_title("第一编 总则") is True
        assert is_part_title("第一章") is False


# ═══════════════════════════════════════════════════════════════
# extract_article_no / extract_chapter_no / extract_section_no
# ═══════════════════════════════════════════════════════════════

class TestExtractFunctions:
    def test_extract_article_no(self):
        assert extract_article_no("第一条 立法目的") == "第一条"
        assert extract_article_no("第十条") == "第十条"
        assert extract_article_no("第一百二十三条") == "第一百二十三条"
        assert extract_article_no("普通文本") is None

    def test_extract_article_no_from_middle(self):
        """从文本中间提取条号。"""
        assert extract_article_no("根据第十条的规定") == "第十条"

    def test_extract_chapter_no(self):
        assert extract_chapter_no("第一章 总则") == "第一章"
        assert extract_chapter_no("普通文本") is None

    def test_extract_section_no(self):
        assert extract_section_no("第一节 基本原则") == "第一节"
        assert extract_section_no("普通文本") is None


# ═══════════════════════════════════════════════════════════════
# build_section_path
# ═══════════════════════════════════════════════════════════════

class TestBuildSectionPath:
    def _make_mock_block(self, text, block_index=0, block_type="paragraph"):
        """创建模拟 block。"""

        class MockBlock:
            def __init__(self, text, block_index, block_type):
                self.text = text
                self.block_id = f"block_{block_index}"
                self.block_index = block_index
                self.block_type = block_type

        return MockBlock(text, block_index, block_type)

    def test_basic_section_path(self):
        """基本章节路径构建。"""
        blocks = [
            self._make_mock_block("第一章 总则"),
            self._make_mock_block("第一条 立法目的"),
            self._make_mock_block("本法是为了..."),
            self._make_mock_block("第二条 适用范围"),
            self._make_mock_block("本法适用于..."),
        ]
        result = build_section_path(blocks)
        assert len(result) == 5

        # 第一章标题自身
        assert result[0].section_path == "第一章 总则"
        assert result[0].law_level == LawLevel.CHAPTER

        # 第一条在 "第一章 总则" 下
        assert result[1].section_path == "第一章 总则"
        assert result[1].article_no == "第一条"
        assert result[1].law_level == LawLevel.ARTICLE

        # 普通段落继承当前章节路径
        assert result[2].section_path == "第一章 总则"
        assert result[2].law_level is None

        # 第二条也在同一章下
        assert result[3].section_path == "第一章 总则"
        assert result[3].article_no == "第二条"

    def test_chapter_switch(self):
        """章切换时路径应更新。"""
        blocks = [
            self._make_mock_block("第一章 总则"),
            self._make_mock_block("第一条 内容"),
            self._make_mock_block("第二章 分则"),
            self._make_mock_block("第十条 具体规定"),
        ]
        result = build_section_path(blocks)

        assert result[0].section_path == "第一章 总则"
        assert result[1].section_path == "第一章 总则"
        assert result[2].section_path == "第二章 分则"
        assert result[3].section_path == "第二章 分则"

    def test_section_within_chapter(self):
        """章内节的路径。"""
        blocks = [
            self._make_mock_block("第一章 总则"),
            self._make_mock_block("第一节 基本原则"),
            self._make_mock_block("第一条 内容"),
        ]
        result = build_section_path(blocks)

        # Block 0: 章标题，此时只有章路径
        assert result[0].section_path == "第一章 总则"
        assert result[0].law_level == LawLevel.CHAPTER

        # Block 1: 节标题，此时路径为 "章 / 节"
        assert result[1].section_path == "第一章 总则 / 第一节 基本原则"
        assert result[1].law_level == LawLevel.SECTION

        # Block 2: 条文，继承章/节路径
        assert result[2].section_path == "第一章 总则 / 第一节 基本原则"
        assert result[2].law_level == LawLevel.ARTICLE

    def test_part_level(self):
        """编层级路径。"""
        blocks = [
            self._make_mock_block("第一编 总则"),
            self._make_mock_block("第一章 基本原则"),
            self._make_mock_block("第一条 内容"),
        ]
        result = build_section_path(blocks)

        assert "第一编 总则" in result[0].section_path
        assert result[0].law_level == LawLevel.PART
        assert "第一章 基本原则" in result[1].section_path

    def test_no_structure(self):
        """无结构标记的文本应正常处理。"""
        blocks = [
            self._make_mock_block("这是一段普通的文本。"),
            self._make_mock_block("这也是普通文本。"),
        ]
        result = build_section_path(blocks)

        assert len(result) == 2
        # 没有章节路径
        for sb in result:
            assert sb.law_level is None
            assert sb.article_no is None

    def test_items_in_article(self):
        """条下各项的识别。"""
        blocks = [
            self._make_mock_block("第一条 行政处罚的种类："),
            self._make_mock_block("（一）警告；"),
            self._make_mock_block("（二）罚款；"),
            self._make_mock_block("（三）没收违法所得。"),
        ]
        result = build_section_path(blocks)

        assert result[0].law_level == LawLevel.ARTICLE
        assert result[1].law_level == LawLevel.ITEM
        assert result[2].law_level == LawLevel.ITEM
        assert result[3].law_level == LawLevel.ITEM

    def test_result_are_structured_blocks(self):
        """返回的应是 StructuredBlock 实例。"""
        blocks = [self._make_mock_block("第一条 测试"), self._make_mock_block("正文内容")]
        result = build_section_path(blocks)

        assert isinstance(result[0], StructuredBlock)
        assert isinstance(result[1], StructuredBlock)

    def test_block_attributes_preserved(self):
        """原始 block 的属性应被保留。"""
        blocks = [self._make_mock_block("第一条 测试", block_index=5, block_type="paragraph")]
        result = build_section_path(blocks)

        assert result[0].block_index == 5
        assert result[0].block_type == "paragraph"

    def test_split_multiple_articles_in_one_block(self):
        """一个 parsed block 内多条“第X条”应拆成多个 structured blocks。"""
        block = self._make_mock_block(
            "第十四条 鼓励国家工作人员率先献血。\n"
            "第十五条 国家机关应当动员适龄公民参加献血。\n"
            "第十六条 公民可以参加单位组织的献血。",
            block_index=8,
        )

        split = split_articles_in_block(block)
        assert len(split) == 3
        assert split[0].text.startswith("第十四条")
        assert split[1].text.startswith("第十五条")
        assert split[2].text.startswith("第十六条")

        result = build_section_path([self._make_mock_block("第三章 献 血", 7), block])
        article_blocks = [sb for sb in result if sb.law_level == LawLevel.ARTICLE]
        assert len(article_blocks) == 3
        assert [sb.article_no for sb in article_blocks] == ["第十四条", "第十五条", "第十六条"]
        assert all(sb.section_path == "第三章 献 血" for sb in article_blocks)

    def test_plain_paragraph_inherits_current_article_no(self):
        """条文后的普通段落应继承当前 article_no。"""
        blocks = [
            self._make_mock_block("第一条 为了规范管理，制定本条例。", 1),
            self._make_mock_block("由房产行政管理部门依法代管的房产，收益由其代为保管。", 2),
        ]

        result = build_section_path(blocks)

        assert result[0].article_no == "第一条"
        assert result[1].law_level is None
        assert result[1].article_no == "第一条"

    def test_item_inherits_current_article_no(self):
        """“四、...”这类项应继承当前 article_no。"""
        blocks = [
            self._make_mock_block("第三条 土地管理应当遵循下列规定：", 1),
            self._make_mock_block("四、未在产权登记期限内办理土地使用权登记手续的，依法查处。", 2),
        ]

        result = build_section_path(blocks)

        assert result[1].law_level == LawLevel.ITEM
        assert result[1].article_no == "第三条"

    def test_new_chapter_resets_current_article_no(self):
        """进入新章后，章标题后的普通段落不应继承上一章最后一条。"""
        blocks = [
            self._make_mock_block("第一章 总则", 1),
            self._make_mock_block("第一条 第一章内容。", 2),
            self._make_mock_block("第二章 管理职责", 3),
            self._make_mock_block("本章规定相关管理职责。", 4),
        ]

        result = build_section_path(blocks)

        assert result[1].article_no == "第一条"
        assert result[2].article_no is None
        assert result[3].section_path == "第二章 管理职责"
        assert result[3].article_no is None


# ═══════════════════════════════════════════════════════════════
# detect_and_annotate_blocks
# ═══════════════════════════════════════════════════════════════

class TestDetectAndAnnotateBlocks:
    def test_is_alias(self):
        """detect_and_annotate_blocks 应与 build_section_path 结果一致。"""

        class MockBlock:
            def __init__(self, text):
                self.text = text
                self.block_id = "test"
                self.block_index = 0
                self.block_type = "paragraph"

        blocks = [
            MockBlock("第一章 总则"),
            MockBlock("第一条 内容"),
        ]
        result1 = build_section_path(blocks)
        result2 = detect_and_annotate_blocks(blocks)

        assert len(result1) == len(result2)
        for a, b in zip(result1, result2):
            assert a.section_path == b.section_path
            assert a.law_level == b.law_level
            assert a.article_no == b.article_no


# ═══════════════════════════════════════════════════════════════
# detect_law_level_for_block
# ═══════════════════════════════════════════════════════════════

class TestDetectLawLevelForBlock:
    def test_with_parsed_block_like_object(self):
        """兼容 ParsedBlock 对象。"""

        class ParsedBlockLike:
            def __init__(self, text):
                self.text = text

        block = ParsedBlockLike("第一条 立法目的")
        assert detect_law_level_for_block(block) == LawLevel.ARTICLE

    def test_with_structured_block(self):
        """兼容 StructuredBlock 对象。"""
        block = StructuredBlock(
            block_id="test",
            text="第一章 总则",
        )
        assert detect_law_level_for_block(block) == LawLevel.CHAPTER


# ═══════════════════════════════════════════════════════════════
# structured_block_to_dict
# ═══════════════════════════════════════════════════════════════

class TestStructuredBlockToDict:
    def test_serialization(self):
        sb = StructuredBlock(
            block_id="test_id",
            text="第一条 测试",
            law_level=LawLevel.ARTICLE,
            section_path="第一章 总则",
            article_no="第一条",
            block_index=3,
            block_type="paragraph",
        )
        d = structured_block_to_dict(sb)
        assert d["block_id"] == "test_id"
        assert d["text"] == "第一条 测试"
        assert d["law_level"] == "条"
        assert d["section_path"] == "第一章 总则"
        assert d["article_no"] == "第一条"
        assert d["block_index"] == 3
        assert d["block_type"] == "paragraph"

    def test_serialization_none_values(self):
        sb = StructuredBlock(block_id="t", text="普通文本")
        d = structured_block_to_dict(sb)
        assert d["law_level"] is None
        assert d["section_path"] is None
        assert d["article_no"] is None
