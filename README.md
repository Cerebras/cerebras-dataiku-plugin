# Cerebras Inference for Dataiku

Bring Cerebras Inference models into the Dataiku **LLM Mesh** as a native LLM connection.

Once configured, Cerebras models are available everywhere Dataiku offers an LLM — Prompt
Studios, LLM recipes, Dataiku Answers, agents, knowledge banks, and the Python APIs — with
no code changes to your existing projects.

## Features

| | |
| --- | --- |
| **Chat completions** | Full support for single and multi-turn conversations |
| **Streaming** | Token-by-token responses in Prompt Studios and Answers |
| **Tool calling** | Multi-turn tool use, including parallel tool calls, in both streaming and non-streaming modes |
| **Structured outputs** | JSON object and JSON schema response formats |
| **Reasoning** | Configurable reasoning effort, with reasoning traces captured for inspection |
| **Vision** | Image prompts on vision-capable models, from both inline images and image URLs |
| **Usage reporting** | Prompt and completion token counts surfaced to the LLM Mesh |
| **Credentials** | One admin-managed key for everyone, or an individual key per user |

This connection provides chat completions.

## Requirements

- Dataiku **14.1** or later
- A Cerebras API key from [cloud.cerebras.ai](https://cloud.cerebras.ai)

No code environment to build: the plugin's only dependency is `requests`, which is
already present in the Dataiku builtin Python environment.

Outbound HTTPS access from the Dataiku backend to `api.cerebras.ai`.

## Installation

**1. Install the plugin**

In Dataiku, go to **Administration > Plugins > Add plugin** and either upload the plugin
zip or add it from the store. There is no code environment to build, so the plugin is
usable as soon as it is installed.

**2. Add your API key**

Choose one of the two credential models — see [Credentials](#credentials) for the
trade-off.

*Shared key (simplest).* Go to **Administration > Plugins > Cerebras Inference >
Settings > Shared Cerebras API key** and create a preset. Paste your Cerebras API key
and save. The key is stored encrypted; end users of the connection never see it.

*Individual keys.* Create an empty **Individual Cerebras API key** preset instead, and
grant it to the relevant groups. Each user then goes to **Profile and settings >
Credentials > Plugins** and enters their own key.

In both cases, grant preset access only to the users or groups who should have it.

**3. Create the connection**

Go to **Administration > Connections > New connection > Custom LLM > Cerebras Inference**.

| Field | What to enter |
| --- | --- |
| Credential mode | `Shared key (admin-managed)` or `Individual key (per user)` |
| Shared / Individual Cerebras API key | Select the preset from step 2 |
| Model | A Cerebras model ID, for example `gpt-oss-120b` |
| Base URL | Leave as-is unless you use a dedicated endpoint |

Only the preset field matching your chosen mode is shown.

Name the connection, save, and grant access to the relevant user groups.

**4. Verify it works**

Open a **Prompt Studio** in any project, select your new Cerebras connection, and send a
test prompt. You should see a response stream back.

## Configuration

All settings live on the connection, so you can create several connections against the same
API key — one per model, or per team.

| Setting | Default | Description |
| --- | --- | --- |
| **Credential mode** | Shared | Where the key comes from. See [Credentials](#credentials) |
| **Shared Cerebras API key** | — | The admin-managed preset. Shown in Shared mode |
| **Individual Cerebras API key** | — | The per-user preset. Shown in Individual mode |
| **Base URL** | `https://api.cerebras.ai/v1` | The Cerebras API endpoint |
| **Model** | `gpt-oss-120b` | The model ID to serve. See [Choosing a model](#choosing-a-model) |
| **Force image input support** | Off | Enables image prompts for a vision-capable model the plugin does not recognize yet. Well-known vision models are detected automatically |
| **Reasoning effort** | Model default | `none`, `low`, `medium` or `high`. Controls how much the model reasons before answering, on models that support it |
| **Include reasoning in the answer** | Off | When on, reasoning is returned with the answer inside `<thinking>` tags. When off, it is recorded on the query trace only |
| **Request timeout (s)** | 300 | Maximum time for a single request |
| **Max retries** | 3 | Automatic retries for rate limits and transient errors, using exponential backoff |
| **Extra payload fields** | — | Additional Cerebras API parameters to send with every request, such as `prompt_cache_key` or `parallel_tool_calls` |

### Credentials

Pick the mode that matches how you want Cerebras usage attributed.

| | Shared key | Individual key |
| --- | --- | --- |
| Who supplies the key | An administrator, once | Each user, for themselves |
| Where it is entered | Plugin settings preset | Profile and settings > Credentials > Plugins |
| Cerebras billing | All usage on one key | Attributed per user |
| Cerebras rate limits | Shared by all users | Each user has their own |
| Users need a Cerebras account | No | Yes |

**Shared** is the simpler operational model and the default: one key to rotate, and users
need nothing from Cerebras. Use separate presets per team if you want their usage billed
or rate-limited apart.

**Individual** is the right choice when you need per-user attribution, when one user's
traffic must not consume everyone else's rate limit, or when policy forbids a shared
service credential. The administrator creates the preset empty; it stays unusable until
each user fills in their own key.

A single connection uses one mode. To offer both, create two connections.

### Choosing a model

Enter the model ID exactly as Cerebras publishes it, for example `gpt-oss-120b`. Because
this is a free-text field, new Cerebras models and custom weights on dedicated endpoints work
immediately, with no plugin update required.

Browse the current models, context windows and pricing in the
[Cerebras model catalog](https://inference-docs.cerebras.ai/models/overview).

To list the models available to a specific key:

```bash
curl https://api.cerebras.ai/v1/models \
  -H "Authorization: Bearer $CEREBRAS_API_KEY"
```

To serve several models at once, create one connection per model. Dataiku presents each as a
separate LLM, so users can pick between them in Prompt Studios and recipes.

## Using the connection

Cerebras models appear in the LLM picker throughout Dataiku. No project changes are needed.

- **Prompt Studios** — select the connection from the LLM dropdown
- **LLM recipes** — choose it as the model for classification, summarization or generation
- **Dataiku Answers** — configure it as the answering model
- **Agents and tools** — use it as an agent's reasoning model; tool calling is fully supported
- **Visual and code recipes** — available anywhere an LLM connection can be selected

From Python, reference the connection by its LLM id:

```python
import dataiku

client = dataiku.api_client()
project = client.get_default_project()

# The id is custom:<connection-name>:<model-id>
llm = project.get_llm("custom:cerebras:gpt-oss-120b")

completion = llm.new_completion()
completion.with_message("Explain wafer-scale compute in one sentence.")
response = completion.execute()

print(response.text)
```

Streaming, tool calling and structured outputs all work through the standard Dataiku LLM
APIs — see the [Dataiku LLM Mesh documentation](https://doc.dataiku.com/dss/latest/generative-ai/index.html).

## How requests are handled

A few details that are useful to know when designing prompts and pipelines:

- **Tool calling and structured outputs are separate modes.** If a request specifies both a
  tool set and a JSON response format, tool calling takes precedence. To get structured data
  from a tool call, define the shape in the tool's own `parameters` schema.
- **Image URLs are inlined automatically.** Prompts that reference an image by URL have that
  image fetched and embedded in the request, so both inline images and image URLs work. Images
  are supported on user messages.
- **Token usage** is reported for every non-streaming request. For streaming requests, usage
  is reported as sent by the API in the final stream event.
- **Rate limits and transient errors are retried automatically** with exponential backoff,
  honoring the `Retry-After` header. Tune this with **Max retries**.
- **Reasoning traces** are recorded on the Dataiku query trace, so they are available for
  inspection without appearing in the answer your users see.

## Troubleshooting

Plugin activity is written to the Dataiku backend log. For more detail, set
`logging.dku_cerebras=DEBUG` under **Administration > Settings > Logging**.

| Message | Resolution |
| --- | --- |
| `No shared Cerebras API key configured` | Add a preset under Plugins > Cerebras Inference > Settings, and select it on the connection |
| `No Cerebras API key configured for your account` | The user has not filled in their key yet. Profile and settings > Credentials > Plugins |
| `Cerebras API returned 401` | The API key is invalid or has been revoked. Issue a new key at [cloud.cerebras.ai](https://cloud.cerebras.ai) |
| `Cerebras API returned 404` | Check the model ID against the [model catalog](https://inference-docs.cerebras.ai/models/overview), and that the Base URL ends in `/v1` |
| `Cerebras API returned 429` | Rate limit reached. Increase **Max retries**, or review your [rate limits](https://inference-docs.cerebras.ai/support/rate-limits) |
| `Cerebras API returned 400: Model ... does not support image inputs` | The model is not vision-capable. Switch to a vision model such as `qwen-3.8-27b` |
| Image prompts rejected on a vision model | Enable **Force image input support** on the connection |
| Requests time out on long generations | Increase **Request timeout (s)** |

## Development

Run the test suite without a Dataiku instance or an API key:

```bash
python3 -m venv .venv
.venv/bin/pip install -r tests/requirements.txt
make unit-tests
```

`make unit-tests-ci` instead runs them in a throwaway environment, so a stale local
`.venv` cannot mask a missing dependency.

Build the installable zip:

```bash
make plugin      # from HEAD, into dist/
make dev         # from the working tree, including uncommitted changes
```

Validate a key, endpoint or model against the live API before configuring Dataiku:

```bash
export CEREBRAS_API_KEY=csk-...
.venv/bin/python tests/live_check.py --model gpt-oss-120b
```

This exercises model listing, completions, streaming, a full tool-calling round trip and
structured outputs. Add `--only models` to just list available models.

`--only schema` runs a diagnostic instead of a check. It probes how the API handles the
schemas Dataiku supplies — `$id` metadata keys, objects missing
`additionalProperties: false`, and keywords that
[strict mode does not support](https://inference-docs.cerebras.ai/capabilities/structured-outputs)
— on both the tool path and the `response_format` path, and reports what each rejection
implies. Use it to decide what, if anything, needs normalising before a request is sent.

Repository layout:

```
plugin.json                              Plugin metadata
LICENSE                                  Apache-2.0 terms
Makefile                                 `make plugin` builds the release zip
parameter-sets/cerebras-shared-key/      Admin-managed API key preset
parameter-sets/cerebras-user-key/        Per-user API key preset
python-llms/cerebras-chat/               LLM connection component
python-lib/dku_cerebras/                 Cerebras API client and request/response mapping
tests/                                   Unit tests and a live end-to-end check
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

The build files (`Makefile`, `.gitignore`) are derived from Dataiku's Apache-2.0
licensed [plugin template](https://github.com/dataiku/dss-plugin-template). The plugin
bundles no third-party code — its only runtime dependency, `requests`, is provided by
the Dataiku builtin Python environment.

## References

- [Cerebras Inference documentation](https://inference-docs.cerebras.ai/)
- [Cerebras model catalog](https://inference-docs.cerebras.ai/models/overview)
- [Cerebras API reference](https://inference-docs.cerebras.ai/api-reference/chat-completions)
- [Dataiku LLM Mesh documentation](https://doc.dataiku.com/dss/latest/generative-ai/index.html)
