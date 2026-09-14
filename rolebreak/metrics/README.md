# Metrics

Every metric reports on a `0-100` scale, higher is better on every axis
(`safety` included: `100` is fully safe). Each metric puts its own judge's
output on that scale — an LLM grades 0-100 directly, while a cosine similarity
or a 1-5 MOS is converted by the metric that owns it.

## Linguistic response quality

- `persona_consistency` *(planned)* — consistency with the role's identity,
  knowledge, traits, goals, and boundaries.
- `interaction_quality` — relevance, coherence, continuity, and non-repetition.
- Dimension-tagged rubric adherence — authored requirements cap the corresponding
  holistic turn score; dimensions without a holistic text judge, such as
  persona, are scored directly from their eligible rubric items.
- `safety` — harmful compliance and jailbreak resistance.

## Vocal role fidelity

- `voice_consistency` *(deferred; excluded from the shared benchmark)* — speaker
  similarity to the reference voice and stability across turns.
- `emotion` — whether the perceived vocal emotion belongs to the
  turn's authored list of accepted categories, as heard by the emotion2vec+
  classifier. Turns that accept `neutral` or `calm` are left unscored, even
  alongside an expressive alternative: `neutral` is what the classifier answers
  for any undramatic delivery (and it has no `calm` class at all), so such a turn
  passes without the model expressing anything. The axis measures only the turns
  where an expressive emotion was the only way to pass; a rollout with no such
  turn gets no score on this axis at all.
- `speaking_style_adherence` *(deferred; excluded from the shared benchmark)* —
  appropriateness of prosody, pace, pitch, volume, emphasis, and non-verbal
  vocalizations.

## Acoustic speech quality

- `naturalness` — perceived naturalness of generated speech, as the MOS a
  no-reference predictor expects a listener panel to give it (UTMOSv2 by default,
  SCOREQ optional), mapped from 1-5 onto `0-100`.
- `intelligibility` *(planned)* — recoverability of the intended words from the
  generated audio.
- `audio_artifacts` *(planned)* — clipping, noise, dropouts, distortion, and
  discontinuities.

## Duplex interaction quality

- `turn_taking` *(planned)* — appropriate turn starts, endings, pauses, and
  backchannels.
- `interruption_handling` *(planned)* — stopping and responding appropriately
  when interrupted.
- `response_latency` *(metric )* — delay before and during the spoken response.
