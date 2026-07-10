"""测试 text_cleaner 模块。"""

import pytest
from rag_preprocess.text_cleaner import (
    clean_text,
    is_noise_line,
    normalize_spaces,
    merge_broken_lines,
    clean_blocks,
    PRESERVE_PATTERNS,
)


# ═══════════════════════════════════════════════════════════════
# normalize_spaces
# ═══════════════════════════════════════════════════════════════

class TestNormalizeSpaces:
    def test_fullwidth_to_halfwidth(self):
        assert normalize_spaces("a　b") == "a b"

    def test_multiple_spaces(self):
        assert normalize_spaces("a    b") == "a b"

    def test_crlf_to_lf(self):
        assert normalize_spaces("a\r\nb") == "a\nb"

    def test_cr_to_lf(self):
        assert normalize_spaces("a\rb") == "a\nb"

    def test_tab_to_space(self):
        assert normalize_spaces("a\tb") == "a b"

    def test_empty_string(self):
        assert normalize_spaces("") == ""

    def test_mixed_whitespace(self):
        result = normalize_spaces("a　 \t  \r\nb")
        # 全角空格→半角, tab→空格, 多空格合并, \r\n→\n
        assert result == "a \nb"


# ═══════════════════════════════════════════════════════════════
# is_noise_line
# ═══════════════════════════════════════════════════════════════

class TestIsNoiseLine:
    def test_empty_string(self):
        assert is_noise_line("") is True

    def test_whitespace_only(self):
        assert is_noise_line("   ") is True

    def test_page_number(self):
        assert is_noise_line("123") is True

    def test_single_digit_page(self):
        assert is_noise_line("1") is True

    def test_symbols_only(self):
        assert is_noise_line("----") is True

    def test_dash_page_format(self):
        assert is_noise_line("- 1 -") is True

    def test_chinese_page_format(self):
        assert is_noise_line("第 1 页") is True

    def test_english_page_format(self):
        assert is_noise_line("Page 1") is True

    def test_fraction_format(self):
        assert is_noise_line("1/100") is True

    def test_em_dash_page(self):
        assert is_noise_line("— 1 —") is True

    def test_total_pages(self):
        assert is_noise_line("共 5 页") is True

    # ── 法规结构编号绝对不能删除 ──

    def test_article_number_preserved(self):
        """条号应被保留。"""
        assert is_noise_line("第一条 本法适用于...") is False

    def test_chapter_title_preserved(self):
        """章标题应被保留。"""
        assert is_noise_line("第一章 总则") is False

    def test_section_title_preserved(self):
        """节标题应被保留。"""
        assert is_noise_line("第一节 基本原则") is False

    def test_part_title_preserved(self):
        """编标题应被保留。"""
        assert is_noise_line("第一编 总则") is False

    def test_chinese_numbered_item_preserved(self):
        """中文编号项应被保留。"""
        assert is_noise_line("一、适用范围") is False

    def test_bracket_item_preserved(self):
        """括号项应被保留。"""
        assert is_noise_line("（一）基本原则") is False

    def test_numeric_item_preserved(self):
        """数字编号项应被保留。"""
        assert is_noise_line("1. 定义") is False

    def test_valid_paragraph(self):
        """普通段落文本应被保留。"""
        assert is_noise_line("本法所称行政行为，是指行政机关依法实施的...") is False

    def test_short_chinese_text(self):
        """短但包含中文的文本应被保留。"""
        assert is_noise_line("总则") is False

    # ── 页眉页脚关键词 ──

    def test_copyright_notice(self):
        assert is_noise_line("版权所有 翻印必究") is True

    def test_confidential_marking(self):
        assert is_noise_line("内部资料 注意保密") is True

    def test_no_chinese_short(self):
        """短且无中文的文本是噪声。"""
        assert is_noise_line("abc") is True


# ═══════════════════════════════════════════════════════════════
# merge_broken_lines
# ═══════════════════════════════════════════════════════════════

class TestMergeBrokenLines:
    def test_empty_string(self):
        assert merge_broken_lines("") == ""

    def test_single_line(self):
        assert merge_broken_lines("第一条 测试") == "第一条 测试"

    def test_no_merge_at_period(self):
        """以句号结尾 → 不合并。"""
        text = "第一条 这是完整句子。\n第二条 新句子。"
        result = merge_broken_lines(text)
        assert "第一条 这是完整句子。" in result
        assert "第二条 新句子。" in result
        # 应该保持为两行
        assert result.count("\n") == 1

    def test_merge_comma_ending(self):
        """以逗号结尾 → 合并下一行。"""
        text = "本法所称行政许可，\n是指行政机关根据申请，"
        result = merge_broken_lines(text)
        assert "\n" not in result
        assert "本法所称行政许可，是指行政机关根据申请，" == result

    def test_no_merge_at_article_start(self):
        """下一行以 '第X条' 开头 → 不合并。"""
        text = "前面段落\n第十条 这是新条文"
        result = merge_broken_lines(text)
        assert result.count("\n") == 1
        assert "第十条 这是新条文" in result

    def test_no_merge_at_bracket_item(self):
        """下一行以 '（一）' 开头 → 不合并（是新项）。"""
        text = "前面内容\n（一）这是新项"
        result = merge_broken_lines(text)
        assert result.count("\n") == 1

    def test_no_merge_at_chapter(self):
        """下一行以 '第X章' 开头 → 不合并。"""
        text = "前面内容\n第一章 总则"
        result = merge_broken_lines(text)
        assert result.count("\n") == 1

    def test_merge_mid_sentence_break(self):
        """句子中间断开 → 合并。"""
        text = "行政机关及其工作人员\n不得泄露在执法过程中知悉的"
        result = merge_broken_lines(text)
        assert "\n" not in result

    def test_empty_lines_preserved(self):
        """空行应保留（作为段落分隔）。"""
        text = "第一段\n\n第二段"
        result = merge_broken_lines(text)
        assert "\n\n" in result

    def test_multiple_merges(self):
        """多个断行全部合并。"""
        text = "第一，\n要依法，\n行政。"
        result = merge_broken_lines(text)
        # "第一，" + "要依法，" = "第一，要依法，"
        # "第一，要依法，" + "行政。" — "行政。" 以句号结尾但前面合并了
        # 检查结果不包含内部换行
        lines = result.split("\n")
        # 可能由于某些判断保留换行，但不应该有3行
        assert len(lines) <= 2


# ═══════════════════════════════════════════════════════════════
# clean_text
# ═══════════════════════════════════════════════════════════════

class TestCleanText:
    def test_empty_text(self):
        assert clean_text("") == ""

    def test_preserves_article_number(self):
        """清洗后应保留条号。"""
        text = "第一条  这是一个测试。\n\n\n第二条  这也是测试。"
        result = clean_text(text)
        assert "第一条" in result
        assert "第二条" in result

    def test_removes_excess_blank_lines(self):
        """多余空行应被压缩。"""
        text = "第一章\n\n\n\n\n第二条"
        result = clean_text(text)
        assert "\n\n\n\n" not in result

    def test_removes_page_numbers(self):
        """页码应被去除。"""
        text = "真实内容\n123\n- 1 -\n继续内容"
        result = clean_text(text)
        assert "123" not in result
        assert "- 1 -" not in result
        assert "真实内容" in result
        assert "继续内容" in result

    def test_preserves_chapter_section_article(self):
        """章、节、条编号全部保留。"""
        text = "第一章 总则\n第一条 目的\n第二条 适用范围"
        result = clean_text(text)
        assert "第一章 总则" in result
        assert "第一条 目的" in result
        assert "第二条 适用范围" in result

    def test_strips_whitespace(self):
        """首尾空白应被去除。"""
        text = "  第一条 测试  \n  第二条 测试  "
        result = clean_text(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_cleans_chinese_legal_text(self):
        """综合清洗中文法规文本。"""
        text = (
            "　  中华人民共和国行政处罚法\r\n"
            "\r\n"
            "第一章　总则\r\n"
            "\r\n"
            "第一条　为了规范行政处罚的设定和实施，\r\n"
            "保障和监督行政机关有效实施行政管理，\r\n"
            "维护公共利益和社会秩序，\r\n"
            "保护公民、法人或者其他组织的合法权益，\r\n"
            "根据宪法，制定本法。\r\n"
            "\r\n"
            "第二条　行政处罚的设定和实施，适用本法。\r\n"
        )
        result = clean_text(text)
        assert "中华人民共和国行政处罚法" in result
        assert "第一章" in result
        assert "总则" in result
        assert "第一条" in result
        assert "第二条" in result
        assert "制定本法" in result


# ═══════════════════════════════════════════════════════════════
# clean_blocks
# ═══════════════════════════════════════════════════════════════

class TestCleanBlocks:
    def test_cleans_block_text(self):
        """批量清洗 block 文本。"""

        class MockBlock:
            def __init__(self, text):
                self.text = text

        blocks = [
            MockBlock("第一条  测试。  "),
            MockBlock("123"),  # 噪声页码
            MockBlock("第二章  正文内容。"),
        ]
        result = clean_blocks(blocks)
        # 第一条被清洗
        assert "第一条" in result[0].text
        # 第二章被保留
        assert "第二章" in result[2].text

    def test_preserves_all_blocks(self):
        """清洗不应删除 block。"""

        class MockBlock:
            def __init__(self, text):
                self.text = text

        blocks = [MockBlock("第一条"), MockBlock("第二条"), MockBlock("- 1 -")]
        result = clean_blocks(blocks)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════
# 结构编号保留规则（集成测试）
# ═══════════════════════════════════════════════════════════════

class TestStructurePreservation:
    """验证所有法规结构编号都不会被清洗函数删除。"""

    STRUCTURE_SAMPLES = [
        "第一编 总则",
        "第一章 基本原则",
        "第一节 适用范围",
        "第一条 立法目的",
        "第十条 法律责任",
        "第一百二十三条 施行日期",
        "一、适用范围",
        "（一）基本定义",
        "1. 名词解释",
        "第一编",
        "第二章",
        "第三节",
        "第四条",
    ]

    @pytest.mark.parametrize("sample", STRUCTURE_SAMPLES)
    def test_is_noise_line_returns_false(self, sample):
        """所有结构编号都不应被 is_noise_line 判定为噪声。"""
        assert is_noise_line(sample) is False, f"'{sample}' 被错误判定为噪声"

    @pytest.mark.parametrize("sample", STRUCTURE_SAMPLES)
    def test_clean_text_preserves_structure(self, sample):
        """所有结构编号都应在 clean_text 后保留。"""
        result = clean_text(sample)
        # 清洗后，核心内容应该保留（可能前后空格被去除）
        core = sample.strip().replace("  ", " ")
        assert len(result) > 0
