whisper_full_get_* getters do not check segment, token or VAD-segment indices

For transparency: AI was used to help analyze and write up this issue. Found while binding the result API in [inferna](https://github.com/shakfu/inferna).

**Environment:** whisper.cpp `v1.9.4` (`927cfce`); unchanged on `master` (`60c0be6`). Line numbers below are from `master`.

**Evidence:** from source reading.

## Summary

The result getters index `std::vector`s with no bounds check, for example (`src/whisper.cpp:8217`, `8329`, `5355`):

```cpp
const char * whisper_full_get_segment_text_from_state(struct whisper_state * state, int i_segment) {
    return state->result_all[i_segment].text.c_str();
}
int64_t whisper_full_get_vad_segment_t0_from_state(struct whisper_state * state, int i) {
    return state->vad_segments[i].orig_start;
}
float whisper_vad_segments_get_segment_t0(struct whisper_vad_segments * segments, int i_segment) {
    return segments->data[i_segment].start;
}
```

The same holds for the `t0/t1`, `no_speech_prob`, `speaker_turn_next`, `n_tokens` and token getters, with and without `_from_state`. An index one past the end reads outside the vector. This matters most for language bindings, where an off-by-one in user code becomes memory corruption instead of an error.

## Suggested fix

Check indices against the corresponding count and return a sentinel (`NULL`, `-1`, `0.0f`, `false`) when out of range, or document that callers must check. A check costs one comparison per call.
