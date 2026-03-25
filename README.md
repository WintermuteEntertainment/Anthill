# Anthill

The Ants Go Marching One by One...

**Anthill** is an end-to-end system for turning real ChatGPT conversations into high-quality instruction datasets and fine-tuned language models.

It consists of four modular components:

- **Anthill Spider** (`anthill-spider/`) — Chrome extension that extracts ChatGPT conversations
- **Anthill Collector** (`anthill-collector/`) — Chrome extension for collecting and packaging conversation data
- **Anthill Loom** (`anthill-loom/`) — Python pipeline that converts conversations into training-ready datasets
- **Anthill Forge** (`anthill-forge/`) — Model training scripts for fine-tuning instruction-following LLMs
