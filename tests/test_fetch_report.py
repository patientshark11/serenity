import os
import sys
import types
from unittest.mock import MagicMock, patch
from pyairtable.formulas import match

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub fpdf to allow importing backend without the dependency
if "fpdf" not in sys.modules:
    stub = types.ModuleType("fpdf")
    class _FPDF:
        pass
    stub.FPDF = _FPDF
    sys.modules["fpdf"] = stub

import backend


def test_fetch_report_uses_safe_formula():
    os.environ["AIRTABLE_API_KEY"] = "key"
    os.environ["AIRTABLE_BASE_ID"] = "base"
    mock_table = MagicMock()
    mock_table.all.return_value = [{"fields": {"Content": "data"}}]
    with patch.object(backend, "Table", return_value=mock_table):
        name = "O'Brian \"Special\""
        backend.fetch_report(name)
        args, kwargs = mock_table.all.call_args
        assert kwargs["formula"] == match({"ReportName": name})
