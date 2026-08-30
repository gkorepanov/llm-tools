# llm-tools

Small shared package for Voicebot's streamed LLM calls.

## Included

- LiteLLM-based async chat streaming;
- bounded initial-request and mid-stream retries;
- ordered fallback across configured model providers;
- empty-response and context-window error handling;
- input/output token accounting and price estimation;
- message conversion helpers;
- UI translation utilities.

Production model chains are defined in the bot's private configuration and
constructed by `bot/parsing/generator.py`. The current chains use Google Gemini
as primary and OpenAI GPT-5.4 mini as fallback.

This package no longer uses LangChain.
