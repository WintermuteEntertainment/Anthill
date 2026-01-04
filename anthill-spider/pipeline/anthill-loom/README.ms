# Anthill

The Ants Go Marching One by One...

# Anthill Loom

Anthill Loom converts raw ChatGPT conversation exports into structured datasets
for training instruction-following language models.

### Inputs
- `chatgpt_conversations_export.json`

### Outputs
- `pairs.jsonl`
- (optional) Hugging Face datasets

### Responsibilities
- Extract user → assistant turn pairs
- Deduplicate and filter low-quality data
- Preserve traceability to source conversations

Loom enforces dataset schemas and performs all data normalization.