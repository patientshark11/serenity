import pytest
import base64
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


def test_generate_and_save_report_saves_name_and_pdf(monkeypatch):
    table = DummyTable()

    def generator_func():
        return "Report body"

    result = generate_and_save_report(table, "Sample Person", generator_func)

    assert result is True
    assert table.upserts, "batch_upsert was not called"
    fields = table.upserts[0][0][0]["fields"]

    assert fields["Name"] == backend.sanitize_name("Sample Person")
    assert "PDF" in fields
    attachment = fields["PDF"][0]
    assert attachment["filename"] == f"{backend.sanitize_name('Sample Person')}.pdf"
    assert attachment["contentType"] == "application/pdf"
    assert isinstance(attachment["data"], str)
    decoded = base64.b64decode(attachment["data"])
    expected_pdf = backend.create_pdf("Report body", summary="Sample Person")
    assert decoded == expected_pdf


class FailingTable:
    def batch_upsert(self, records, key_fields=None):
        raise Exception("upsert failed")


def test_generate_and_save_report_raises_on_upsert_failure():
    table = FailingTable()

    def generator_func():
        return "Report body"

    with pytest.raises(Exception, match="upsert failed"):
        generate_and_save_report(table, "Sample Person", generator_func)


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
            pass

        def table(self, base_id, table_name):
            return DummyTable()

    monkeypatch.setattr(sync_reports, "Api", DummyApi)

    people = sync_reports.get_key_people()

    assert captured["formula"] == "FIND('person', {Entity Type})"
    assert people == ["Kim", "Diego"]
