import backend
from sync_reports import generate_and_save_report


class DummyTable:
    def __init__(self):
        self.upserts = []

    def batch_upsert(self, records, key_fields=None):
        self.upserts.append((records, key_fields))


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
    assert isinstance(attachment["data"], (bytes, bytearray))
