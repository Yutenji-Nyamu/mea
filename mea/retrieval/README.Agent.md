# README.Agent: minimal task-source retrieval

The RoboTwin `envs/` directory contains 50 public task implementations plus
framework files. The retrieval agent receives only the task-name catalog, the
user request, the canonical task name, and the validated VariantSpec.

Select one to three task source files:

1. The canonical task must always be selected first as the authoritative
   implementation.
2. Select an additional task only when its name strongly suggests a useful
   implementation pattern.
3. For color or appearance changes, `blocks_ranking_rgb` is a useful example.
4. Do not select framework files or invent task names.
5. Retrieval chooses reference source files; it does not change the canonical
   task identity, checkpoint, success criterion, or evaluation request.

Return strict JSON with `selected_tasks` and a concise `reasoning` string.
