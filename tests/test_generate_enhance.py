"""Unit tests for patches/generate_enhance.py."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test: patches/generate_enhance.py
# ---------------------------------------------------------------------------

class TestGenerateEnhance:
    """Test generate_enhance wrapper logic using mocks."""

    @dataclass
    class FakeSTTOutput:
        text: str = ""
        language: str = None

    def test_language_defaults_to_chinese(self):
        """When language not passed, defaults to Chinese."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput(text="你好")

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            # Re-import to get patched version
            from patches.generate_enhance import _generate_patched
            result = _generate_patched(MagicMock(), "audio.wav")

        assert captured_kwargs["language"] == "Chinese"
        assert result.language == "Chinese"

    def test_language_explicit_preserved(self):
        """Explicit language kwarg is not overridden."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput(text="hello")

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            result = _generate_patched(MagicMock(), "audio.wav", language="English")

        assert captured_kwargs["language"] == "English"
        assert result.language == "English"

    def test_repetition_penalty_injected(self):
        """repetition_penalty is injected from default when not provided."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput()

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            _generate_patched(MagicMock(), "audio.wav")

        assert "repetition_penalty" in captured_kwargs
        assert captured_kwargs["repetition_penalty"] > 1.0

    def test_repetition_penalty_explicit_preserved(self):
        """Explicit repetition_penalty is not overridden."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput()

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            _generate_patched(MagicMock(), "audio.wav", repetition_penalty=1.5)

        assert captured_kwargs["repetition_penalty"] == 1.5

    def test_language_backfill_on_output(self):
        """STTOutput.language is set to the actual language used."""
        fake_output = self.FakeSTTOutput(text="test")
        assert fake_output.language is None  # before patch

        def mock_generate(self_, audio, **kwargs):
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            result = _generate_patched(MagicMock(), "audio.wav")

        assert result.language == "Chinese"


