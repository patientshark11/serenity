import backend


def test_sanitize_name_removes_disallowed_characters():
    raw = "Report/Name's \"Test\""
    assert backend.sanitize_name(raw) == "ReportNames Test"


def test_create_pdf_returns_bytes():
    result = backend.create_pdf("Hello")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_fetch_report_returns_content(mock_airtable):
    content = backend.fetch_report("Any")
    assert content == "Mocked content"
