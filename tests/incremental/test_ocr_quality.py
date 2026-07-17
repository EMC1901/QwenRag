from rag_preprocess.incremental.parsers.ocr_quality import assess_page, document_failed
def test_fixed_ocr_boundaries():
    assert assess_page([('甲'*25,.79)]).status=='warning'
    assert assess_page([('甲'*25,.59)]).status=='severe'
    assert document_failed([assess_page([('甲'*25,.59)])])
