import os
import pytest
import backend


def test_usage_tracker_initialization():
    """Test that usage tracker initializes with default values."""
    tracker = backend.OpenAIUsageTracker()
    assert tracker.embedding_calls == 0
    assert tracker.completion_calls == 0
    assert tracker.max_embeddings == 500  # default
    assert tracker.max_completions == 50  # default


def test_usage_tracker_custom_limits(monkeypatch):
    """Test that usage tracker respects environment variables."""
    monkeypatch.setenv("MAX_EMBEDDINGS_PER_RUN", "100")
    monkeypatch.setenv("MAX_COMPLETIONS_PER_RUN", "25")
    
    tracker = backend.OpenAIUsageTracker()
    assert tracker.max_embeddings == 100
    assert tracker.max_completions == 25


def test_track_embedding_call():
    """Test embedding call tracking."""
    tracker = backend.OpenAIUsageTracker()
    tracker.reset_counters()
    
    tracker.track_embedding_call()
    assert tracker.embedding_calls == 1
    
    tracker.track_embedding_call()
    assert tracker.embedding_calls == 2


def test_track_completion_call():
    """Test completion call tracking."""
    tracker = backend.OpenAIUsageTracker()
    tracker.reset_counters()
    
    tracker.track_completion_call()
    assert tracker.completion_calls == 1
    
    tracker.track_completion_call()
    assert tracker.completion_calls == 2


def test_embedding_limit_exceeded():
    """Test that embedding limit is enforced."""
    tracker = backend.OpenAIUsageTracker()
    tracker.max_embeddings = 2
    tracker.reset_counters()
    
    # First two calls should work
    tracker.track_embedding_call()
    tracker.track_embedding_call()
    
    # Third call should raise RuntimeError
    with pytest.raises(RuntimeError, match="Embedding API limit reached"):
        tracker.track_embedding_call()


def test_completion_limit_exceeded():
    """Test that completion limit is enforced."""
    tracker = backend.OpenAIUsageTracker()
    tracker.max_completions = 2
    tracker.reset_counters()
    
    # First two calls should work
    tracker.track_completion_call()
    tracker.track_completion_call()
    
    # Third call should raise RuntimeError
    with pytest.raises(RuntimeError, match="Completion API limit reached"):
        tracker.track_completion_call()


def test_reset_counters():
    """Test that reset_counters works properly."""
    tracker = backend.OpenAIUsageTracker()
    
    # Add some counts
    tracker.track_embedding_call()
    tracker.track_completion_call()
    
    assert tracker.embedding_calls == 1
    assert tracker.completion_calls == 1
    
    # Reset and verify
    tracker.reset_counters()
    assert tracker.embedding_calls == 0
    assert tracker.completion_calls == 0


def test_get_usage_tracker_returns_singleton():
    """Test that get_usage_tracker returns the same instance."""
    tracker1 = backend.get_usage_tracker()
    tracker2 = backend.get_usage_tracker()
    
    assert tracker1 is tracker2


def test_get_embedding_tracks_usage(monkeypatch):
    """Test that get_embedding function tracks usage."""
    # Mock OpenAI client response
    class MockResponse:
        def __init__(self):
            self.data = [type('obj', (object,), {'embedding': [0.1, 0.2, 0.3]})]
    
    class MockOpenAIClient:
        def __init__(self):
            self.embeddings = self
        
        def create(self, **kwargs):
            return MockResponse()
    
    tracker = backend.get_usage_tracker()
    tracker.reset_counters()
    
    # Call get_embedding and verify tracking
    result = backend.get_embedding("test text", MockOpenAIClient())
    
    assert tracker.embedding_calls == 1
    assert result == [0.1, 0.2, 0.3]


def test_get_embedding_respects_limit(monkeypatch):
    """Test that get_embedding respects usage limits."""
    class MockOpenAIClient:
        def __init__(self):
            self.embeddings = self
        
        def create(self, **kwargs):
            raise AssertionError("Should not be called when limit exceeded")
    
    tracker = backend.get_usage_tracker()
    tracker.max_embeddings = 1
    tracker.reset_counters()
    
    # Set the counter to the limit
    tracker.embedding_calls = 1
    
    # Next call should fail before reaching OpenAI
    with pytest.raises(RuntimeError, match="Embedding API limit reached"):
        backend.get_embedding("test text", MockOpenAIClient())