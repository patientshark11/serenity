import types
import sys
from unittest.mock import patch
from pyairtable.formulas import match

# Provide a dummy fpdf module so backend imports without the optional dependency
fpdf_module = types.ModuleType("fpdf")
fpdf_module.FPDF = object
sys.modules.setdefault("fpdf", fpdf_module)

import backend


def test_fetch_report_with_special_characters(monkeypatch):
    report_name = "He said \"Hello\" & Goodbye's #1"

    monkeypatch.setenv("AIRTABLE_API_KEY", "key")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "base")

    expected_formula = match({"ReportName": report_name})

    class FakeTable:
        def __init__(self, api_key, base_id, table_name):
            assert table_name == "GeneratedReports"
        def all(self, formula=None, max_records=None):
            assert formula == expected_formula
            return [{"fields": {"Content": "special report content"}}]

    with patch.object(backend, "Table", FakeTable):
        content = backend.fetch_report(report_name)

    assert content == "special report content"
