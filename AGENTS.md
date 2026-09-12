# Codex project instructions

## Workflow
- Never commit directly to `main`.
- Make changes in a separate task branch and propose them through a pull request.
- Keep each pull request focused on one task.
- Do not merge a pull request unless the repository owner has reviewed the changes.

## Research integrity
- Do not change the research methodology, model specification, dependent variable, explanatory variables, sample definition, or statistical interpretation unless explicitly requested.
- Do not delete, overwrite, or transform raw source data unless explicitly requested.
- Clearly distinguish data-cleaning changes from methodological changes.
- Prefer reproducible and deterministic processing steps.

## Repository hygiene
- Do not commit raw datasets, credentials, API keys, local machine paths, virtual environments, IDE metadata, caches, logs, or generated temporary files unless explicitly requested.
- Preserve a clear separation between source code, data inputs, outputs, and documentation.

## Python code structure
- Keep major sections clearly separated: imports, paths, settings, model switches, variable definitions, data loading, data preparation, model specification, estimation, diagnostics, and outputs.
- Put user-adjustable settings and `True`/`False` model switches near the top of the relevant script or configuration file.
- Use descriptive variable names.
- Keep comments concise and explain intent or methodological reasoning rather than restating obvious syntax.
- Preserve the existing language of comments and documentation unless asked to change it.

## Validation and review
- Before proposing changes, run the relevant code or tests when possible.
- If required data or dependencies are unavailable, state clearly what was not tested.
- Do not claim that results were reproduced unless they were actually reproduced.
- When reviewing code, explicitly check for logical inconsistencies, incorrect variable definitions, reversed event/censoring indicators, data leakage, incorrect merges, unit mismatches, and discrepancies between variable documentation and implementation.
- Flag suspected methodological or interpretation problems for the repository owner instead of silently changing the research design.
