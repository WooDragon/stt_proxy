"""Unit tests for patches/__init__.py (apply_all)."""



# ---------------------------------------------------------------------------
# Test: patches/__init__.py
# ---------------------------------------------------------------------------

class TestApplyAll:
    """Test that apply_all() loads all patches without error."""

    def test_apply_all_idempotent(self):
        """Calling apply_all multiple times doesn't crash."""
        from patches import apply_all
        apply_all()
        apply_all()  # second call should be harmless


