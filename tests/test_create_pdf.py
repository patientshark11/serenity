from pathlib import Path
import sys
from importlib.util import find_spec
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

if find_spec("fpdf") is None:
    pytest.skip("fpdf is required for this test", allow_module_level=True)

from backend import create_pdf


def test_create_pdf_returns_bytes():
    """create_pdf should return a non-empty bytes object"""
    pdf_bytes = create_pdf("Hello World")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
