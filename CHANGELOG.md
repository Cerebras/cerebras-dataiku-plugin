# Changelog

## Version 1.0.0 - Initial release - 2026-09-08

- Cerebras Inference chat completions as a Dataiku LLM Mesh custom LLM connection
- Streaming and non-streaming completions
- Tool calling, including parallel tool calls, in both modes
- Structured outputs via JSON object and JSON schema response formats
- Configurable reasoning effort, with reasoning traces recorded on the query trace
- Image inputs on vision-capable models, from both inline images and image URLs
- Prompt and completion token usage reported to the LLM Mesh
- Automatic retries on rate limits and transient errors, honouring `Retry-After`
- Free-text model ID, so new Cerebras models work without a plugin update
- Shared (admin-managed) or individual (per-user) API keys, selected per connection
- Extra payload fields for Cerebras parameters without a dedicated setting
- No code environment to build: `requests` is already in the DSS builtin environment
