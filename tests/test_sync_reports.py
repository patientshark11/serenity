import pytest
import backend
import sync_reports
from sync_reports import generate_and_save_report


class DummyTable:
    def __init__(self):
        self.upserts = []

    def batch_upsert(self, records, key_fields=None):
        self.upserts.append((records, key_fields))
        # Simulate Airtable returning a created or updated record
        return [{"id": "rec1", "fields": records[0]["fields"]}]


def test_generate_and_save_report_generates_pdf_without_uploading(monkeypatch):
    table = DummyTable()

    captured = {}

    def fake_create_pdf(content, summary):
        captured["content"] = content
        captured["summary"] = summary
        return b"pdf-bytes"

    monkeypatch.setattr(backend, "create_pdf", fake_create_pdf)

    def generator_func():
        return "Report body"

    result = generate_and_save_report(table, "Sample Person", generator_func)

    assert result is True
    assert table.upserts, "batch_upsert was not called"
    fields = table.upserts[0][0][0]["fields"]

    assert fields["Name"] == backend.sanitize_name("Sample Person")
    assert set(fields.keys()) == {"Name", "Content", "LastGenerated"}
    assert "PDF" not in fields
    assert captured == {
        "content": "Report body",
        "summary": "Sample Person",
    }


def test_generate_and_save_report_respects_custom_name_field(monkeypatch):
    table = DummyTable()

    def generator_func():
        return "Report body"

    monkeypatch.setenv("AIRTABLE_REPORT_NAME_FIELD", "ReportLabel")

    generate_and_save_report(table, "Custom Person", generator_func)

    records, key_fields = table.upserts[0]
    fields = records[0]["fields"]

    assert fields["ReportLabel"] == backend.sanitize_name("Custom Person")
    assert "Name" not in fields
    assert key_fields == ["ReportLabel"]
    assert set(fields.keys()) == {"ReportLabel", "Content", "LastGenerated"}
    assert "PDF" not in fields


def test_generate_and_save_report_payload_contains_only_supported_fields():
    table = DummyTable()

    def generator_func():
        return "Another report"

    generate_and_save_report(table, "Another Person", generator_func)

    records, key_fields = table.upserts[0]
    fields = records[0]["fields"]
    assert set(fields.keys()) == {"Name", "Content", "LastGenerated"}
    assert "PDF" not in fields


class FailingTable:
    def batch_upsert(self, records, key_fields=None):
        raise Exception("upsert failed")


def test_generate_and_save_report_raises_on_upsert_failure():
    table = FailingTable()

    def generator_func():
        return "Report body"

    with pytest.raises(Exception, match="upsert failed"):
        generate_and_save_report(table, "Sample Person", generator_func)


def test_generate_and_save_report_supports_streaming_chunks():
    table = DummyTable()

    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.delta = FakeDelta(content)
            self.message = None

    class FakeChunk:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    def generator_func():
        def _stream():
            yield FakeChunk("Hello ")
            yield FakeChunk("world")
            yield FakeChunk(None)

        return _stream()

    generate_and_save_report(table, "Streamed", generator_func)

    fields = table.upserts[0][0][0]["fields"]
    assert fields["Name"] == backend.sanitize_name("Streamed")
    assert fields["Content"] == "Hello world"


def test_get_key_people_returns_airtable_results(monkeypatch):
    monkeypatch.setenv("AIRTABLE_API_KEY", "key")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "base")
    monkeypatch.setenv("AIRTABLE_TABLE_NAME", "table")

    captured = {}

    class DummyTable:
        def __init__(self, *args, **kwargs):
            pass

        def all(self, formula=None):
            captured["formula"] = formula
            return [
                {"fields": {"Name": "Kim"}},
                {"fields": {"Name": "Diego"}},
                {"fields": {"Name": "Kim"}},  # duplicate
            ]

    class DummyApi:
        def __init__(self, api_key):
            self.closed = False

        def table(self, base_id, table_name):
            return DummyTable()

        def close(self):
            self.closed = True

    monkeypatch.setattr(sync_reports, "Api", DummyApi)

    people = sync_reports.get_key_people()

    assert captured["formula"] == "FIND('person', {Entity Type})"
    assert people == ["Kim", "Diego"]
