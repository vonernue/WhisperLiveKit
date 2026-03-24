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


class TokensAlignment:

    def __init__(self, state: Any, args: Any, sep: Optional[str]) -> None:
        self.state = state
        self.diarization = args.diarization

        self.all_tokens: List[ASRToken] = []
        self.all_diarization_segments: List[SpeakerSegment] = []
        self.all_translation_segments: List[Any] = []

        self.new_tokens: List[ASRToken] = []
        self.new_diarization: List[SpeakerSegment] = []
        self.new_translation: List[Any] = []
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

        self._retention_seconds: float = _DEFAULT_RETENTION_SECONDS

    def update(self) -> None:
        """Drain state buffers into the running alignment context."""
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)
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

        idx = _find_cutoff(self.validated_segments)
        if idx:
            self.validated_segments = self.validated_segments[idx:]

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

    def build_token_speaker_segments(
        self,
        diarization_segments: List[SpeakerSegment],
    ) -> Tuple[List[Segment], str]:
        """Split transcript using token-level speaker overlap instead of punctuation."""
        diarization_buffer = ''
        if not diarization_segments:
            return self.compute_punctuations_segments(), diarization_buffer

        segments: List[Segment] = []
        pending_tokens: List[ASRToken] = []
        pending_speaker: Optional[int] = None
        last_diarization_end = diarization_segments[-1].end

        def flush_pending() -> None:
            nonlocal pending_tokens, pending_speaker
            if not pending_tokens or pending_speaker is None:
                pending_tokens = []
                pending_speaker = None
                return
            segment = self._segment_from_token_group(pending_tokens, pending_speaker)
            if segment is not None:
                segments.append(segment)
            pending_tokens = []
            pending_speaker = None

        for token in self.all_tokens:
            if token.is_silence():
                flush_pending()
                silence_segment = Segment.from_tokens([token], is_silence=True)
                if silence_segment is not None:
                    segments.append(silence_segment)
                continue

            if token.start >= last_diarization_end:
                diarization_buffer += token.text
                continue

            speaker = self._speaker_for_token(token, diarization_segments)
            if speaker is None:
                diarization_buffer += token.text
                continue

            if pending_speaker is None or speaker == pending_speaker:
                pending_tokens.append(token)
                pending_speaker = speaker
            else:
                flush_pending()
                pending_tokens.append(token)
                pending_speaker = speaker

        flush_pending()
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
