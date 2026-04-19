import logging
from time import time
from typing import Any, List, Optional, Tuple, Union

from whisperlivekit.timed_objects import (
    ASRToken,
    PuncSegment,
    Segment,
    Silence,
    SilentSegment,
    SpeakerSegment,
    TimedText,
)

_DEFAULT_RETENTION_SECONDS: float = 300.0
_ASR_SEGMENT_MAX_BACKWARD_SNAP_SECONDS: float = 1.75
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TokensAlignment:

    def __init__(self, state: Any, args: Any, sep: Optional[str]) -> None:
        self.state = state
        self.diarization = args.diarization

        self.all_tokens: List[ASRToken] = []
        self.all_diarization_segments: List[SpeakerSegment] = []
        self.all_translation_segments: List[Any] = []
        self.all_asr_segment_ends: List[float] = []

        self.new_tokens: List[ASRToken] = []
        self.new_diarization: List[SpeakerSegment] = []
        self.new_translation: List[Any] = []
        self.new_asr_segment_ends: List[float] = []
        self.new_translation_buffer: Union[TimedText, str] = TimedText()
        self.new_tokens_buffer: List[Any] = []
        self.sep: str = sep if sep is not None else ' '
        self.beg_loop: Optional[float] = None

        self.validated_segments: List[Segment] = []
        self.current_line_tokens: List[ASRToken] = []
        self.diarization_buffer: List[ASRToken] = []

        self.last_punctuation = None
        self.last_uncompleted_punc_segment: PuncSegment = None
        self.unvalidated_tokens: PuncSegment = []
        self._logged_split_events: set[tuple] = set()

        self._retention_seconds: float = _DEFAULT_RETENTION_SECONDS

    def update(self) -> None:
        """Drain state buffers into the running alignment context."""
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_asr_segment_ends, self.state.new_asr_segment_ends = self.state.new_asr_segment_ends, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)
        for seg_end in self.new_asr_segment_ends:
            if not self.all_asr_segment_ends or abs(self.all_asr_segment_ends[-1] - seg_end) > 0.05:
                self.all_asr_segment_ends.append(seg_end)
        self.new_translation_buffer = self.state.new_translation_buffer

    def _prune(self) -> None:
        """Drop tokens/segments older than ``_retention_seconds`` from the latest token."""
        if not self.all_tokens:
            return

        latest = self.all_tokens[-1].end
        cutoff = latest - self._retention_seconds
        if cutoff <= 0:
            return

        def _find_cutoff(items: list) -> int:
            """Return the index of the first item whose end >= cutoff."""
            for i, item in enumerate(items):
                if item.end >= cutoff:
                    return i
            return len(items)

        idx = _find_cutoff(self.all_tokens)
        if idx:
            self.all_tokens = self.all_tokens[idx:]

        idx = _find_cutoff(self.all_diarization_segments)
        if idx:
            self.all_diarization_segments = self.all_diarization_segments[idx:]

        idx = _find_cutoff(self.all_translation_segments)
        if idx:
            self.all_translation_segments = self.all_translation_segments[idx:]

        while self.all_asr_segment_ends and self.all_asr_segment_ends[0] < cutoff:
            self.all_asr_segment_ends.pop(0)

        idx = _find_cutoff(self.validated_segments)
        if idx:
            self.validated_segments = self.validated_segments[idx:]

        if self.all_tokens:
            latest = self.all_tokens[-1].end
            cutoff_logged = latest - self._retention_seconds
            if cutoff_logged > 0:
                self._logged_split_events = {
                    event for event in self._logged_split_events if event[0] >= cutoff_logged
                }

    def _log_split_once(self, event: tuple, message: str, *args: Any) -> None:
        """Emit split diagnostics only once per unique boundary decision."""
        if event in self._logged_split_events:
            return
        self._logged_split_events.add(event)
        logger.info(message, *args)

    def add_translation(self, segment: Segment) -> None:
        """Append translated text segments that overlap with a segment."""
        if segment.translation is None:
            segment.translation = ''
        for ts in self.all_translation_segments:
            if ts.is_within(segment):
                if ts.text:
                    segment.translation += ts.text + self.sep
            elif segment.translation:
                break


    def compute_punctuations_segments(self, tokens: Optional[List[ASRToken]] = None) -> List[PuncSegment]:
        """Group tokens into segments split by punctuation and explicit silence."""
        segments = []
        segment_start_idx = 0
        for i, token in enumerate(self.all_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i+1],
                    )
                    segments.append(segment)
                    segment_start_idx = i+1

        final_segment = PuncSegment.from_tokens(
            tokens=self.all_tokens[segment_start_idx:],
        )
        if final_segment:
            segments.append(final_segment)
        return segments

    def compute_new_punctuations_segments(self) -> List[PuncSegment]:
        new_punc_segments = []
        segment_start_idx = 0
        self.unvalidated_tokens += self.new_tokens
        for i, token in enumerate(self.unvalidated_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    new_punc_segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                new_punc_segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i+1],
                    )
                    new_punc_segments.append(segment)
                    segment_start_idx = i+1

        self.unvalidated_tokens = self.unvalidated_tokens[segment_start_idx:]
        return new_punc_segments


    def concatenate_diar_segments(self) -> List[SpeakerSegment]:
        """Merge consecutive diarization slices that share the same speaker."""
        if not self.all_diarization_segments:
            return []
        merged = [
            SpeakerSegment(
                start=self.all_diarization_segments[0].start,
                end=self.all_diarization_segments[0].end,
                speaker=self.all_diarization_segments[0].speaker,
            )
        ]
        for segment in self.all_diarization_segments[1:]:
            if segment.speaker == merged[-1].speaker:
                merged[-1].end = segment.end
            else:
                merged.append(
                    SpeakerSegment(
                        start=segment.start,
                        end=segment.end,
                        speaker=segment.speaker,
                    )
                )
        return merged


    @staticmethod
    def intersection_duration(seg1: TimedText, seg2: TimedText) -> float:
        """Return the overlap duration between two timed segments."""
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def _speaker_for_token(
        self,
        token: ASRToken,
        diarization_segments: List[SpeakerSegment],
    ) -> Optional[int]:
        """Return the 1-based speaker id with the largest overlap for a token."""
        max_overlap = 0.0
        max_overlap_speaker: Optional[int] = None
        for diarization_segment in diarization_segments:
            intersec = self.intersection_duration(token, diarization_segment)
            if intersec > max_overlap:
                max_overlap = intersec
                max_overlap_speaker = diarization_segment.speaker + 1
        return max_overlap_speaker

    @staticmethod
    def _segment_from_token_group(tokens: List[ASRToken], speaker: int) -> Optional[Segment]:
        """Build a text segment from tokens and attach the resolved speaker id."""
        segment = Segment.from_tokens(tokens)
        if segment is None:
            return None
        segment.speaker = speaker
        return segment

    def _find_segment_snap_index(
        self,
        tokens: List[ASRToken],
        segment_start_idx: int,
        boundary_time: float,
        raw_change_idx: int,
    ) -> Tuple[int, float, bool]:
        """Return the best token split index near a diarization boundary.

        Uses a deterministic backward-only cascade:
        - find the latest ASR segment end at or before the diarization cut
        - prefer the closest backward token boundary with a positive gap
        - otherwise use the closest backward token boundary
        - fall back to the raw split if no ASR boundary is available
        """
        raw_boundary = tokens[raw_change_idx - 1].end
        target_boundary: Optional[float] = None
        for seg_end in self.all_asr_segment_ends:
            if seg_end > boundary_time:
                break
            if boundary_time - seg_end <= _ASR_SEGMENT_MAX_BACKWARD_SNAP_SECONDS:
                target_boundary = seg_end

        if target_boundary is None:
            return raw_change_idx, raw_boundary, False

        best_with_gap: Optional[Tuple[float, float, float, int, float]] = None
        best_any: Optional[Tuple[float, float, int, float]] = None

        for i in range(segment_start_idx + 1, raw_change_idx + 1):
            candidate_end = tokens[i - 1].end
            candidate_start = tokens[i].start
            backward_delta = boundary_time - candidate_end
            if backward_delta > _ASR_SEGMENT_MAX_BACKWARD_SNAP_SECONDS:
                continue

            target_delta = abs(candidate_end - target_boundary)
            if best_any is None or (target_delta, backward_delta) < best_any[:2]:
                best_any = (target_delta, backward_delta, i, candidate_end)

            gap = candidate_start - candidate_end
            if gap > 0.05:
                candidate = (target_delta, -gap, backward_delta, i, candidate_end)
                if best_with_gap is None or candidate < best_with_gap:
                    best_with_gap = candidate

        if best_with_gap is not None:
            _, _, _, split_idx, split_boundary = best_with_gap
            if split_idx != raw_change_idx:
                return split_idx, split_boundary, True
            return raw_change_idx, raw_boundary, False

        if best_any is not None:
            _, _, split_idx, split_boundary = best_any
            if split_idx != raw_change_idx:
                return split_idx, split_boundary, True
        return raw_change_idx, raw_boundary, False

    def _build_speech_segments(
        self,
        tokens: List[ASRToken],
        speakers: List[int],
    ) -> List[Segment]:
        """Build speaker-attributed segments by correcting each raw speaker-change boundary."""
        if not tokens:
            return []

        segments: List[Segment] = []
        segment_start_idx = 0
        current_speaker = speakers[0]

        for raw_change_idx in range(1, len(tokens)):
            next_speaker = speakers[raw_change_idx]
            if next_speaker == current_speaker:
                continue

            split_idx, split_boundary, snapped = self._find_segment_snap_index(
                tokens=tokens,
                segment_start_idx=segment_start_idx,
                boundary_time=tokens[raw_change_idx].start,
                raw_change_idx=raw_change_idx,
            )

            event = (
                round(tokens[raw_change_idx].start, 2),
                current_speaker,
                next_speaker,
                "snap" if snapped else "raw",
            )
            if snapped:
                self._log_split_once(
                    event,
                    "Diarization split snapped to ASR segment boundary: diar=%.2fs split=%.2fs prev_spk=%s next_spk=%s shift=%.2fs",
                    tokens[raw_change_idx].start,
                    split_boundary,
                    current_speaker,
                    next_speaker,
                    tokens[raw_change_idx].start - split_boundary,
                )
            else:
                self._log_split_once(
                    event,
                    "Diarization split used raw boundary: diar=%.2fs prev_spk=%s next_spk=%s pending_tokens=%d",
                    tokens[raw_change_idx].start,
                    current_speaker,
                    next_speaker,
                    raw_change_idx - segment_start_idx,
                )

            segment = self._segment_from_token_group(tokens[segment_start_idx:split_idx], current_speaker)
            if segment is not None:
                segments.append(segment)

            segment_start_idx = split_idx
            current_speaker = next_speaker

        final_segment = self._segment_from_token_group(tokens[segment_start_idx:], current_speaker)
        if final_segment is not None:
            segments.append(final_segment)
        return segments

    def build_token_speaker_segments(
        self,
        diarization_segments: List[SpeakerSegment],
    ) -> Tuple[List[Segment], str]:
        """Split transcript using token-level speaker overlap instead of punctuation."""
        diarization_buffer = ''
        if not diarization_segments:
            return self.compute_punctuations_segments(), diarization_buffer

        segments: List[Segment] = []
        chunk_tokens: List[ASRToken] = []
        chunk_speakers: List[int] = []
        last_diarization_end = diarization_segments[-1].end

        def flush_chunk() -> None:
            nonlocal chunk_tokens, chunk_speakers
            if not chunk_tokens:
                return
            segments.extend(self._build_speech_segments(chunk_tokens, chunk_speakers))
            chunk_tokens = []
            chunk_speakers = []

        for token in self.all_tokens:
            if token.is_silence():
                flush_chunk()
                silence_segment = Segment.from_tokens([token], is_silence=True)
                if silence_segment is not None:
                    segments.append(silence_segment)
                continue

            if token.start >= last_diarization_end:
                flush_chunk()
                diarization_buffer += token.text
                continue

            speaker = self._speaker_for_token(token, diarization_segments)
            if speaker is None:
                flush_chunk()
                diarization_buffer += token.text
                continue

            chunk_tokens.append(token)
            chunk_speakers.append(speaker)

        flush_chunk()
        return segments, diarization_buffer

    def get_lines_diarization(self) -> Tuple[List[Segment], str]:
        """Build segments when diarization is enabled and track overflow buffer."""
        diarization_segments = self.concatenate_diar_segments()
        return self.build_token_speaker_segments(diarization_segments)


    def get_lines(
            self,
            diarization: bool = False,
            translation: bool = False,
            current_silence: Optional[Silence] = None,
            audio_time: Optional[float] = None,
        ) -> Tuple[List[Segment], str, Union[str, TimedText]]:
        """Return the formatted segments plus buffers, optionally with diarization/translation.

        Args:
            audio_time: Current audio stream position in seconds. Used as fallback
                for ongoing silence end time instead of wall-clock (which breaks
                when audio is fed faster or slower than real-time).
        """
        # Fallback for ongoing silence: prefer audio stream time over wall-clock
        _silence_now = audio_time if audio_time is not None else (time() - self.beg_loop)

        if diarization:
            segments, diarization_buffer = self.get_lines_diarization()
        else:
            diarization_buffer = ''
            for token in self.new_tokens:
                if isinstance(token, Silence):
                    if self.current_line_tokens:
                        self.validated_segments.append(Segment.from_tokens(self.current_line_tokens))
                        self.current_line_tokens = []

                    end_silence = token.end if token.has_ended else _silence_now
                    if self.validated_segments and self.validated_segments[-1].is_silence():
                        self.validated_segments[-1].end = end_silence
                    else:
                        self.validated_segments.append(SilentSegment(
                            start=token.start,
                            end=end_silence
                        ))
                else:
                    self.current_line_tokens.append(token)

            segments = list(self.validated_segments)
            if self.current_line_tokens:
                segments.append(Segment.from_tokens(self.current_line_tokens))

        if current_silence:
            end_silence = current_silence.end if current_silence.has_ended else _silence_now
            if segments and segments[-1].is_silence():
                segments[-1] = SilentSegment(start=segments[-1].start, end=end_silence)
            else:
                segments.append(SilentSegment(
                    start=current_silence.start,
                    end=end_silence
                ))
        if translation:
            [self.add_translation(segment) for segment in segments if not segment.is_silence()]

        self._prune()

        return segments, diarization_buffer, self.new_translation_buffer.text
