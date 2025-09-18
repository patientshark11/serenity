import os
import sys
import types
import warnings
import pytest

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Create dummy modules for unavailable dependencies
sys.modules.setdefault("openai", types.ModuleType("openai"))

weaviate_module = types.ModuleType("weaviate")
classes_module = types.ModuleType("classes")
config_module = types.ModuleType("config")
init_module = types.ModuleType("init")

class _Vectorizer:
    @staticmethod
    def none():
        return None

config_module.Configure = types.SimpleNamespace(Vectorizer=_Vectorizer)
config_module.Property = object
config_module.DataType = object
init_module.Auth = object
classes_module.config = config_module
classes_module.init = init_module
weaviate_module.classes = classes_module
sys.modules["weaviate"] = weaviate_module
sys.modules["weaviate.classes"] = classes_module
sys.modules["weaviate.classes.config"] = config_module
sys.modules["weaviate.classes.init"] = init_module

pyairtable_module = types.ModuleType("pyairtable")

class DummyTable:
    def __init__(self, fields=("Name",)):
        self._fields = fields

    def first(self, formula=None):
        return {"fields": {"Content": "Mocked content"}}

    def schema(self):
        class Field:
            def __init__(self, name):
                self.name = name

        class Schema:
            def __init__(self, names):
                self.fields = [Field(n) for n in names]

        return Schema(self._fields)

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

pyairtable_module.Api = DummyApi
sys.modules["pyairtable"] = pyairtable_module

fpdf_module = types.ModuleType("fpdf")


class DummyXPos:
    LMARGIN = object()


class DummyYPos:
    NEXT = object()


class DummyFPDF:
    def add_page(self):
        pass

    def set_font(self, family, *args, **kwargs):
        if family == "Arial":
            warnings.warn("Arial font fallback is deprecated", DeprecationWarning)

    def cell(self, *args, **kwargs):
        if len(args) > 4 or "ln" in kwargs:
            warnings.warn("'ln' argument is deprecated", DeprecationWarning)

    def ln(self, *args, **kwargs):
        pass

    def multi_cell(self, *args, **kwargs):
        pass

    def set_text_color(self, *args, **kwargs):
        pass

    def output(self, *args, **kwargs):
        if "dest" in kwargs:
            warnings.warn("'dest' argument is deprecated", DeprecationWarning)
        return b"PDF"


fpdf_module.FPDF = DummyFPDF
fpdf_module.XPos = DummyXPos
fpdf_module.YPos = DummyYPos
sys.modules["fpdf"] = fpdf_module

import backend

@pytest.fixture
def mock_airtable(monkeypatch):
    monkeypatch.setenv("AIRTABLE_API_KEY", "test-key")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "test-base")
    monkeypatch.setattr(backend, "Api", DummyApi)
