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


def test_map_reduce_query_uses_distinct_models(monkeypatch):
    class DummyItem:
        def __init__(self, chunk):
            self.properties = {"chunk_content": chunk}

    class DummyQuery:
        def __init__(self):
            self.calls = []

        def near_vector(self, **kwargs):
            self.calls.append(kwargs)

            class Response:
                objects = [DummyItem("chunk one"), DummyItem("chunk two")]

            return Response()

    class DummyCollection:
        def __init__(self):
            self.query = DummyQuery()
            self.iterator_called = False

        def iterator(self):
            self.iterator_called = True
            return []

    class DummyCollections:
        def __init__(self, collection):
            self._collection = collection

        def exists(self, name):
            return True

        def get(self, name):
            return self._collection

    class DummyWeaviateClient:
        def __init__(self, collection):
            self.collections = DummyCollections(collection)

    class RecordingCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("stream"):
                return iter(["stream"])

            class Message:
                def __init__(self, content):
                    self.content = content

            class Choice:
                def __init__(self, content):
                    self.message = Message(content)

            class Response:
                def __init__(self, content):
                    self.choices = [Choice(content)]

            return Response("Mapped info")

    class DummyChat:
        def __init__(self, completions):
            self.completions = completions

    class DummyOpenAIClient:
        def __init__(self, completions):
            self.chat = DummyChat(completions)

    captured_embedding = {}

    def fake_embedding(text, client):
        captured_embedding["text"] = text
        return [0.1, 0.2]

    monkeypatch.setattr(backend, "get_embedding", fake_embedding)

    completions = RecordingCompletions()
    openai_client = DummyOpenAIClient(completions)
    collection = DummyCollection()
    weaviate_client = DummyWeaviateClient(collection)

    result = backend._map_reduce_query(
        weaviate_client,
        openai_client,
        map_prompt_template="{chunk_content}",
        reduce_prompt_template="{combined_text}",
        model="reduce-model",
        map_model="map-model",
        fallback_search={"query": "timeline", "limit": 2},
    )

    # Reduce step should stream content when a report is generated
    assert hasattr(result, "__iter__")

    map_calls = [call for call in completions.calls if not call.get("stream")]
    reduce_calls = [call for call in completions.calls if call.get("stream")]

    assert map_calls, "Expected MAP calls to be recorded"
    assert {call["model"] for call in map_calls} == {"map-model"}
    assert len(reduce_calls) == 1
    assert reduce_calls[0]["model"] == "reduce-model"

    assert not collection.iterator_called, "Iterator should not be used for fallback search"
    assert collection.query.calls, "Expected fallback search to issue a query"
    call = collection.query.calls[0]
    assert call["limit"] == 2
    assert call["return_properties"] == ["chunk_content"]
    assert captured_embedding["text"] == "timeline"
