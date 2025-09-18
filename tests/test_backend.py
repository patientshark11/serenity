import warnings
import pytest
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


def test_fetch_report_missing_field(monkeypatch):
    class DummyTable:
        def first(self, formula=None):
            raise AssertionError("should not query when field missing")

        def schema(self):
            class Field:
                def __init__(self, name):
                    self.name = name

            class Schema:
                fields = [Field("Name")]

            return Schema()

    class DummyApi:
        def __init__(self, api_key):
            pass

        def table(self, base_id, table_name):
            return DummyTable()

        def close(self):
            pass

    monkeypatch.setenv("AIRTABLE_API_KEY", "key")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "base")
    monkeypatch.setenv("AIRTABLE_REPORT_NAME_FIELD", "Missing")
    monkeypatch.setattr(backend, "Api", DummyApi)

    with pytest.raises(ValueError) as exc:
        backend.fetch_report("Any")

    assert "Field 'Missing' not found" in str(exc.value)


def test_fetch_report_sanitizes_name_in_formula(monkeypatch):
    captured = {}

    class DummyTable:
        def first(self, formula=None):
            captured["formula"] = formula
            return {"fields": {"Content": "Mocked content"}}

        def schema(self):
            class Field:
                def __init__(self, name):
                    self.name = name

            class Schema:
                fields = [Field("Name")]

            return Schema()

    class DummyApi:
        def __init__(self, api_key):
            self.closed = False

        def table(self, base_id, table_name):
            return DummyTable()

        def close(self):
            self.closed = True

    monkeypatch.setenv("AIRTABLE_API_KEY", "key")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "base")
    monkeypatch.setattr(backend, "Api", DummyApi)

    name = "Report/Name's \"Test\""
    backend.fetch_report(name)

    assert "{Name}='ReportNames Test'" in captured["formula"]
    assert "{Name}='Report/Name\\'s \"Test\"'" in captured["formula"]


def test_fetch_reports_uses_single_api_and_no_resource_warning(monkeypatch):
    monkeypatch.setenv("AIRTABLE_API_KEY", "key")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "base")

    class DummyTable:
        def first(self, formula=None):
            return {"fields": {"Content": "Mocked content"}}

        def schema(self):
            class Field:
                def __init__(self, name):
                    self.name = name

            class Schema:
                fields = [Field("Name")]

            return Schema()

    class DummyApi:
        instances = 0

        def __init__(self, api_key):
            self.closed = False
            DummyApi.instances += 1

        def table(self, base_id, table_name):
            return DummyTable()

        def close(self):
            self.closed = True

        def __del__(self):
            if not self.closed:
                warnings.warn("DummyApi not closed", ResourceWarning)

    monkeypatch.setattr(backend, "Api", DummyApi)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error", ResourceWarning)
        reports = backend.fetch_reports(["A", "B", "C"])

    assert DummyApi.instances == 1
    assert len(reports) == 3
    assert all(content == "Mocked content" for content in reports.values())
    assert len(w) == 0
