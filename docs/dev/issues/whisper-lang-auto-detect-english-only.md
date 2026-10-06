whisper_lang_auto_detect returns a language id for English-only models

For transparency: AI was used to help analyze and write up this issue. Found while binding `whisper_lang_auto_detect` in [inferna](https://github.com/shakfu/inferna).

**Environment:** whisper.cpp `v1.9.4` (`927cfce`); unchanged on `master` (`60c0be6`). Line numbers below are from `master`.

**Evidence:** from source reading.

## Summary

`whisper_lang_auto_detect_with_state` (line 4237) reads the logit of `whisper_token_lang(ctx, id)` for every language (line 4197):

```cpp
const auto token_lang = whisper_token_lang(ctx, kv.second.first);   // sot + 1 + id
logits_id.emplace_back(state->logits[token_lang], kv.second.first);
```

English-only (`*.en`) vocabularies have no language tokens, so `sot + 1 + id` names unrelated tokens. The function returns their softmax as language probabilities and a top id with no meaning. `whisper_full` guards the language prompt with `whisper_is_multilingual(ctx)`; `whisper_lang_auto_detect` does not.

## Suggested fix

Return a negative error from `whisper_lang_auto_detect` / `_with_state` when `!whisper_is_multilingual(ctx)`.
