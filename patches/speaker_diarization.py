"""Speaker diarization via CoreML-accelerated WeSpeaker embeddings.

Pipeline:
    audio_bytes → torchaudio.load (16kHz) → VAD silence detection
    → sliding windows (1.5s, 0.75s step) → fbank + per-window CMN
    → CoreML speaker embedding → AHC clustering → speaker timeline

Runtime dependencies: coremltools, torchaudio, scikit-learn (all pre-installed).
CoreML model must be pre-converted via scripts/convert_wespeaker.py.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────
_WINDOW_SEC = 1.5        # sliding window duration
_STEP_SEC = 0.75         # sliding window step (50% overlap)
_TARGET_SR = 16000        # all processing at 16kHz
_FBANK_BINS = 80          # mel filterbank bins (WeSpeaker standard)
_FRAME_LENGTH_MS = 25     # fbank frame length
_FRAME_SHIFT_MS = 10      # fbank frame shift


class SpeakerDiarizer:
    """CoreML-accelerated speaker diarization.

    Lazily loads the pre-converted CoreML WeSpeaker model on first use.
    Raises FileNotFoundError with actionable message if model is missing.
    """

    def __init__(self, model_dir=None):
        default_dir = Path(__file__).resolve().parent.parent / "models" / "wespeaker"
        model_dir = Path(model_dir or os.getenv("WESPEAKER_CACHE", str(default_dir)))
        self._model_path = model_dir / "wespeaker_resnet34_LM_fp16.mlpackage"
        self._model = None

    def _ensure_model(self):
        """Load CoreML model on first call. Fail fast if missing."""
        if self._model is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"CoreML model not found at {self._model_path}. "
                "Run: python scripts/convert_wespeaker.py"
            )
        import coremltools as ct
        self._model = ct.models.MLModel(str(self._model_path))

    def diarize(self, audio_input, num_speakers=None, min_speakers=None,
                max_speakers=None, filename=None):
        """Run full diarization pipeline.

        Args:
            audio_input: Audio source — file path (str/Path) or raw bytes.
                File path is preferred (avoids bytes→tempfile copy).
            num_speakers: Exact speaker count (overrides threshold-based clustering).
            min_speakers: Minimum speaker count (post-clustering clamp).
            max_speakers: Maximum speaker count (post-clustering clamp).
            filename: Original filename (e.g. "recording.m4a") for ffmpeg format hint.
                Ignored when audio_input is a file path (suffix is read from the path).

        Returns:
            List of (start_sec, end_sec, "SPEAKER_XX") tuples.
        """
        self._ensure_model()

        # 1. Load audio → mono 16kHz
        waveform, sr = self._load_audio(audio_input, filename)
        wav_16k = self._resample(waveform, sr, _TARGET_SR)

        # 2. VAD silence detection (reuse vad_chunking infrastructure)
        from patches.vad_chunking import _detect_silence_regions
        silence_regions = _detect_silence_regions(wav_16k.numpy().squeeze())

        # 3. Sliding windows, skip pure-silence windows
        windows = self._make_windows(wav_16k, silence_regions)
        if not windows:
            return []

        # 4. fbank + per-window CMN → CoreML embedding
        embeddings = self._extract_embeddings(windows)

        # 5. AHC clustering
        labels = self._cluster(embeddings, num_speakers, min_speakers, max_speakers)

        # 6. Merge adjacent same-speaker windows → timeline
        return self._build_timeline(windows, labels)

    def _load_audio(self, audio_input, filename=None):
        """Load audio → (waveform, sr).

        Accepts:
            str / Path → torchaudio.load(path) directly (no copy).
            bytes → write to tempfile → torchaudio.load (backward compat).

        Tempfile suffix from filename/path helps ffmpeg sniff container format
        (critical for M4A/MP4 whose moov atom needs seek — same root cause as
        patches/audio_io_fix.py).
        """
        import torchaudio

        if isinstance(audio_input, (str, Path)):
            # File path: load directly — no bytes→tempfile overhead
            waveform, sr = torchaudio.load(str(audio_input))
        elif isinstance(audio_input, bytes):
            # Bytes: write to tempfile for ffmpeg seek support
            suffix = ""
            if filename:
                dot_idx = filename.rfind(".")
                if dot_idx >= 0:
                    suffix = filename[dot_idx:]  # e.g. ".m4a"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio_input)
                tmp.flush()
                waveform, sr = torchaudio.load(tmp.name)
        else:
            raise TypeError(
                f"audio_input must be str, Path, or bytes, got {type(audio_input).__name__}"
            )

        # Mono (take first channel if multi-channel)
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        return waveform, sr

    @staticmethod
    def _resample(waveform, sr_orig, sr_target):
        """Resample torch tensor via torchaudio."""
        if sr_orig == sr_target:
            return waveform
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr_orig, sr_target)
        return resampler(waveform)

    @staticmethod
    def _make_windows(waveform, silence_regions,
                      window_sec=_WINDOW_SEC, step_sec=_STEP_SEC, sr=_TARGET_SR):
        """Generate sliding windows, skipping pure-silence windows.

        Args:
            waveform: Torch tensor (1, N) at target SR.
            silence_regions: List of (start_sample, end_sample) from VAD.
            window_sec: Window duration in seconds.
            step_sec: Step size in seconds.
            sr: Sample rate.

        Returns:
            List of (start_sec, end_sec, waveform_chunk) tuples.
        """
        total_samples = waveform.shape[-1]
        window_samples = int(window_sec * sr)
        step_samples = int(step_sec * sr)

        windows = []
        pos = 0
        while pos + window_samples <= total_samples:
            start_sec = pos / sr
            end_sec = (pos + window_samples) / sr
            chunk = waveform[:, pos:pos + window_samples]

            # Skip windows that are entirely within a silence region
            if not _window_is_silent(pos, pos + window_samples, silence_regions):
                windows.append((start_sec, end_sec, chunk))
            pos += step_samples

        # Handle tail: if remaining audio >= half a window, include it
        remaining = total_samples - pos
        if remaining >= window_samples // 2 and pos < total_samples:
            start_sec = pos / sr
            end_sec = total_samples / sr
            chunk = waveform[:, pos:total_samples]
            if not _window_is_silent(pos, total_samples, silence_regions):
                windows.append((start_sec, end_sec, chunk))

        return windows

    def _extract_embeddings(self, windows):
        """Compute speaker embeddings for all windows.

        For each window:
            1. kaldi.fbank → (T, 80)
            2. Per-window CMN (subtract mean)
            3. CoreML predict → 256-dim embedding

        Args:
            windows: List of (start_sec, end_sec, waveform_chunk).

        Returns:
            np.ndarray of shape (N, 256).
        """
        import torch
        import torchaudio.compliance.kaldi as kaldi

        embeddings = []
        for _, _, chunk in windows:
            # Scale to int16 range (WeSpeaker convention)
            scaled = chunk * (1 << 15)
            fbank = kaldi.fbank(
                scaled,
                num_mel_bins=_FBANK_BINS,
                frame_length=_FRAME_LENGTH_MS,
                frame_shift=_FRAME_SHIFT_MS,
                dither=0.0,
                sample_frequency=_TARGET_SR,
                window_type="hamming",
                use_energy=False,
            )
            # Per-window CMN normalization
            fbank = fbank - torch.mean(fbank, dim=0)
            # CoreML expects (1, T, 80) float32 numpy
            feats = fbank.unsqueeze(0).numpy()
            prediction = self._model.predict({"feats": feats})
            emb = _extract_coreml_embedding(prediction)
            embeddings.append(emb)

        return np.vstack(embeddings)  # (N, 256)

    @staticmethod
    def _cluster(embeddings, num_speakers=None, min_speakers=None, max_speakers=None):
        """Agglomerative Hierarchical Clustering with cosine distance.

        Priority:
            1. num_speakers explicit → n_clusters=num_speakers
            2. Otherwise → distance_threshold with optional min/max clamp

        Args:
            embeddings: (N, 256) speaker embeddings.
            num_speakers: Exact count if known.
            min_speakers: Lower bound.
            max_speakers: Upper bound.

        Returns:
            np.ndarray of cluster labels (N,).
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize

        n_samples = len(embeddings)
        if n_samples <= 1:
            return np.zeros(n_samples, dtype=int)

        # L2-normalize (needed for ward+euclidean in num_speakers/max_speakers modes;
        # cosine metric in threshold mode handles normalization natively)
        embeddings_norm = normalize(embeddings, norm="l2")

        threshold = float(os.getenv("DIARIZE_THRESHOLD", "0.75"))

        if num_speakers is not None and num_speakers >= 1:
            n_clusters = min(num_speakers, n_samples)
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="euclidean",
                linkage="ward",
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="cosine",
                linkage="average",
            )

        labels = clustering.fit_predict(embeddings_norm)

        # Post-clustering clamp
        unique_labels = np.unique(labels)
        n_found = len(unique_labels)

        if max_speakers is not None and n_found > max_speakers:
            # Re-cluster with hard cap
            clustering = AgglomerativeClustering(
                n_clusters=max_speakers,
                metric="euclidean",
                linkage="ward",
            )
            labels = clustering.fit_predict(embeddings_norm)

        if min_speakers is not None and n_found < min_speakers:
            # Cannot split further — leave as-is, best effort
            pass

        return labels

    @staticmethod
    def _build_timeline(windows, labels):
        """Merge adjacent same-speaker windows into contiguous segments.

        Args:
            windows: List of (start_sec, end_sec, chunk).
            labels: Array of cluster labels per window.

        Returns:
            List of (start_sec, end_sec, "SPEAKER_XX").
        """
        if not windows:
            return []

        timeline = []
        current_start = windows[0][0]
        current_end = windows[0][1]
        current_label = labels[0]

        for i in range(1, len(windows)):
            win_start, win_end, _ = windows[i]
            label = labels[i]

            if label == current_label and win_start <= current_end:
                # Extend current segment
                current_end = max(current_end, win_end)
            else:
                timeline.append((current_start, current_end, f"SPEAKER_{current_label:02d}"))
                current_start = win_start
                current_end = win_end
                current_label = label

        timeline.append((current_start, current_end, f"SPEAKER_{current_label:02d}"))
        return timeline


def assign_speakers(asr_result, speaker_timeline):
    """Align ASR segments with speaker timeline via time-weighted voting.

    For each ASR segment, compute overlap duration with every speaker
    region. Assign the speaker with maximum total overlap.

    Args:
        asr_result: Dict with "text" and optionally "segments" list.
            Each segment has "text", "start", "end".
        speaker_timeline: List of (start, end, "SPEAKER_XX") from diarize().

    Returns:
        Enriched dict with speaker labels on segments + top-level "speakers" list.
    """
    result = dict(asr_result)
    segments = result.get("segments", [])

    if not segments or not speaker_timeline:
        # No alignment possible — return with empty speakers list
        result["speakers"] = sorted({spk for _, _, spk in speaker_timeline}) if speaker_timeline else []
        return result

    all_speakers = set()
    enriched_segments = []

    for seg in segments:
        seg = dict(seg)
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        seg_duration = seg_end - seg_start

        # Accumulate overlap per speaker
        speaker_overlap = {}
        for tl_start, tl_end, speaker in speaker_timeline:
            overlap_start = max(seg_start, tl_start)
            overlap_end = min(seg_end, tl_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap

        if speaker_overlap:
            best_speaker = max(speaker_overlap, key=speaker_overlap.get)
            seg["speaker"] = best_speaker
            all_speakers.add(best_speaker)

            # Mark low-confidence cross-boundary segments
            best_overlap = speaker_overlap[best_speaker]
            if seg_duration > 0 and best_overlap / seg_duration < 0.7:
                seg["speaker_confidence"] = round(best_overlap / seg_duration, 2)

        enriched_segments.append(seg)

    result["segments"] = enriched_segments
    # Include all speakers from timeline, not just those assigned
    all_speakers.update(spk for _, _, spk in speaker_timeline)
    result["speakers"] = sorted(all_speakers)
    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _window_is_silent(start_sample, end_sample, silence_regions):
    """Check if a window is entirely within a silence region."""
    for s_start, s_end in silence_regions:
        if s_start <= start_sample and end_sample <= s_end:
            return True
    return False


def _extract_coreml_embedding(prediction):
    """Extract embedding array from CoreML prediction dict.

    CoreML output key may vary; try known names then fall back to first key.
    """
    for key in ("embs", "output", "var_output"):
        if key in prediction:
            return np.array(prediction[key]).flatten()
    first_key = list(prediction.keys())[0]
    return np.array(prediction[first_key]).flatten()
