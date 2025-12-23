# Context
Goal: Move to template-driven extraction (OCR -> rules). Template defines output fields, types, and validation. API returns template-specific data; no LLM fallback in this step.

# Plan
1) Update template JSON to include field definitions, constraints, and sum row targets.
2) Add template schema validator for types and constraints.
3) Refactor rule extractor to emit template-shaped data and validate it.
4) Simplify pipeline to run OCR -> rules only and return rule result.
5) Update API response schema and route to return template-specific payload.
6) Update tests for rule extraction, pipeline, and template schema.
7) Update README to reflect template-driven output and error behavior.
