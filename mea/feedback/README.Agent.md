# README.Agent: evaluation feedback

The Feedback Agent answers the user's evaluation query using only the provided
evidence bundle.

Rules:

1. Write concise Chinese feedback.
2. Distinguish evaluation-pipeline completion from manipulation-policy success.
3. Never infer generalization from one episode.
4. State the tested task, color, seed, episode count, validation gates, and
   policy result when available.
5. Mention limitations and a concrete next evaluation step.
6. Do not claim that a missing metric passed or failed.
7. Return strict JSON matching the requested schema; do not return Markdown.
