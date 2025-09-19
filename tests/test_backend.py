import types
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


def test_create_pdf_emits_no_deprecation_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = backend.create_pdf(
            "Body",
            summary="Summary",
            sources=[{"title": "Source", "url": "https://example.com"}],
        )

    assert isinstance(result, bytes)
    assert not any(w.category is DeprecationWarning for w in caught)


def test_fetch_report_returns_content(mock_airtable):
    content = backend.fetch_report("Any")
    assert content == "Mocked content"


def test_fetch_report_defaults_to_generate_reports_table(monkeypatch):
    captured = {}

    class DummyTable:
        def first(self, formula=None):
            return {"fields": {"Content": "Mocked content"}}

        def schema(self):
            return {"fields": [{"name": "Name"}]}

    class DummyApi:
        def __init__(self, api_key):
            pass

        def table(self, base_id, table_name):
            captured["base_id"] = base_id
            captured["table_name"] = table_name
            return DummyTable()

        def close(self):
            pass

    monkeypatch.setenv("AIRTABLE_API_KEY", "key")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "base")
    monkeypatch.delenv("AIRTABLE_REPORTS_TABLE_NAME", raising=False)
    monkeypatch.setattr(backend, "Api", DummyApi)

    content = backend.fetch_report("Any")

    assert content == "Mocked content"
    assert captured["base_id"] == "base"
    assert captured["table_name"] == "GenerateReports"


def test_fetch_report_accepts_dict_schema(monkeypatch):
    class DummyTable:
        def first(self, formula=None):
            return {"fields": {"Content": "Mocked content"}}

        def schema(self):
            return {
                "fields": [
                    {"name": "Name"},
                    {"name": "Content"},
                ]
            }

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

    content = backend.fetch_report("Any")

    assert content == "Mocked content"


def test_fetch_report_accepts_object_schema(monkeypatch):
    class Field:
        def __init__(self, name):
            self.name = name

    class DummyTable:
        def first(self, formula=None):
            return {"fields": {"Content": "Mocked content"}}

        def schema(self):
            class Schema:
                fields = [Field("Name"), Field("Content")]

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

    content = backend.fetch_report("Any")

    assert content == "Mocked content"


def test_fetch_report_missing_field(monkeypatch):
    class DummyTable:
        def first(self, formula=None):
            raise AssertionError("should not query when field missing")

        def schema(self):
            return {
                "fields": [
                    {"name": "Name"},
                ],
            }

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
            return {
                "fields": [
                    {"name": "Name"},
                ],
            }

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
            return {
                "fields": [
                    {"name": "Name"},
                ],
            }

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


def test_map_reduce_query_handles_braces(monkeypatch):
    class DummyObject(types.SimpleNamespace):
        pass

    class DummyCollection:
        def __init__(self):
            self.objects = [
                DummyObject(properties={"chunk_content": "Chunk with {braces}"})
            ]

            class DummyQuery:
                def __init__(self, outer):
                    self._outer = outer

                def near_vector(self, near_vector=None, limit=None):
                    return types.SimpleNamespace(objects=self._outer.objects)

            self.query = DummyQuery(self)

        def iterator(self):
            return iter(self.objects)

    class DummyCollections:
        def __init__(self):
            self._collection = DummyCollection()

        def exists(self, name):
            return True

        def get(self, name):
            return self._collection

    class DummyWeaviateClient:
        def __init__(self):
            self.collections = DummyCollections()

    class DummyOpenAI:
        def __init__(self):
            self.prompts = []

            class DummyCompletions:
                def __init__(self, outer):
                    self._outer = outer

                def create(self, *, messages, stream=False, **kwargs):
                    self._outer.prompts.append(messages[-1]["content"])
                    if stream:
                        return [{"choices": [{"delta": {"content": "Final"}}]}]
                    return types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(
                                    content="Mapped {info}"
                                )
                            )
                        ]
                    )

            class DummyChat:
                def __init__(self, outer):
                    self.completions = DummyCompletions(outer)

            self.chat = DummyChat(self)

    monkeypatch.setattr(backend, "get_embedding", lambda *a, **k: [0])

    openai_client = DummyOpenAI()
    weaviate_client = DummyWeaviateClient()

    result = backend._map_reduce_query(
        weaviate_client,
        openai_client,
        "Map: {chunk_content}",
        "Reduce: {combined_text} ({entity_name})",
        model="gpt-4",
        entity_name="Entity {Name}",
    )

    assert openai_client.prompts[0].endswith("Chunk with {braces}")
    assert "Mapped {info}" in openai_client.prompts[-1]
    assert "Entity {Name}" in openai_client.prompts[-1]
    assert result == [{"choices": [{"delta": {"content": "Final"}}]}]
