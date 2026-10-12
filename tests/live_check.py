"""Live end-to-end check against the real Cerebras API, without DSS.

Exercises the same client + mapping code the plugin runs, so you can validate a key,
a base URL or a model before wiring up a DSS connection.

    export CEREBRAS_API_KEY=csk-...
    .venv/bin/python tests/live_check.py                       # default model
    .venv/bin/python tests/live_check.py --model qwen-3.8-27b
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-lib"))

from dku_cerebras.client import CerebrasAPIError, CerebrasClient  # noqa: E402
from dku_cerebras.mapping import (  # noqa: E402
    build_payload,
    parse_completion,
    stream_to_dataiku_chunks,
)
from dku_cerebras.models import model_label  # noqa: E402

# Stand-in for the identity metadata Dataiku may attach to the schemas it hands the
# plugin. The value is irrelevant; only whether the API tolerates the key matters.
SCHEMA_ID = "https://example.invalid/schemas/get_weather"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _client(args):
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        sys.exit("Set CEREBRAS_API_KEY first.")
    return CerebrasClient(api_key=api_key, base_url=args.base_url, timeout=120, max_retries=2)


def check_models(client, args):
    """Verify GET /v1/models, which backs the Model dropdown on the connection form."""
    print("\n== list models ==")
    model_ids = client.list_models()
    for model_id in sorted(model_ids):
        print("  {}".format(model_label(model_id)))
    assert model_ids, "no models returned"
    if args.model not in model_ids:
        print("NOTE: --model {} is not in this account's catalog.".format(args.model))


def check_completion(client, args):
    print("\n== non-streaming ==")
    payload = build_payload(
        args.model,
        {"messages": [{"role": "user", "content": "Reply with exactly: pong"}]},
        {"temperature": 0, "maxOutputTokens": 64},
    )
    result = parse_completion(client.chat_completion(payload))
    result.pop("_reasoning", None)
    print(json.dumps(result, indent=2))
    assert result["text"], "empty completion"


def check_streaming(client, args):
    print("\n== streaming ==")
    payload = build_payload(
        args.model,
        {"messages": [{"role": "user", "content": "Count from 1 to 5, comma separated."}]},
        {"temperature": 0, "maxOutputTokens": 128},
    )
    response = client.stream_chat_completion(payload)
    text, footer = "", None
    for chunk in stream_to_dataiku_chunks(response.iter_lines()):
        if "text" in chunk:
            text += chunk["text"]
            sys.stdout.write(chunk["text"])
            sys.stdout.flush()
        elif "footer" in chunk:
            footer = chunk["footer"]
    print("\nfooter: " + json.dumps(footer))
    assert text, "empty stream"
    if not footer.get("completionTokens"):
        print("NOTE: no usage reported on the stream; DSS will show 0 tokens for streamed calls.")


def check_tool_calling(client, args):
    print("\n== tool calling ==")
    payload = build_payload(
        args.model,
        {"messages": [{"role": "user", "content": "What is the weather in Sunnyvale?"}]},
        {"temperature": 0, "tools": [WEATHER_TOOL], "toolChoice": {"type": "AUTO"}},
    )
    result = parse_completion(client.chat_completion(payload))
    print("finishReason: {}".format(result["finishReason"]))
    print("toolCalls: " + json.dumps(result["toolCalls"], indent=2))
    if result["finishReason"] != "tool_calls":
        print("NOTE: the model answered directly instead of calling the tool.")
        return

    # Feed the result back the way DSS does, to check the tool round-trip.
    call = result["toolCalls"][0]
    followup = build_payload(
        args.model,
        {"messages": [
            {"role": "user", "content": "What is the weather in Sunnyvale?"},
            {"role": "assistant", "toolCalls": result["toolCalls"]},
            {"role": "tool", "toolOutputs": [{"callId": call["id"], "output": "21C and sunny"}]},
        ]},
        {"temperature": 0, "tools": [WEATHER_TOOL]},
    )
    final = parse_completion(client.chat_completion(followup))
    print("final answer: " + final["text"])


def check_structured_output(client, args):
    print("\n== structured output ==")
    payload = build_payload(
        args.model,
        {"messages": [{"role": "user", "content": "Give me the capital of France."}]},
        {"temperature": 0, "responseFormat": {"type": "json", "schema": {
            "type": "object",
            "properties": {"country": {"type": "string"}, "capital": {"type": "string"}},
            "required": ["country", "capital"],
            "additionalProperties": False,
        }}},
    )
    result = parse_completion(client.chat_completion(payload))
    print(result["text"])
    json.loads(result["text"])


def check_schema_metadata(client, args):
    """Probe whether the API tolerates JSON Schema metadata keys such as `$id`.

    The plugin forwards DSS schemas with no inspection, and the two paths differ:
    tools go out as DSS supplied them (no `strict`), while _build_response_format
    always sets strict:true. Per the strict-mode docs that is the stricter contract
    -- additionalProperties:false is required on every object, and the accepted
    metadata keywords are description/title/$schema/default, which does not include
    $id.

    https://inference-docs.cerebras.ai/capabilities/structured-outputs

    Diagnostic only: it reports rather than asserting, because the answers decide
    what build_payload has to normalise.
    """
    print("\n== schema handling ==")

    def attempt(label, tools=None, response_format=None):
        settings = {"temperature": 0, "maxOutputTokens": 64}
        if tools:
            settings["tools"] = tools
        if response_format:
            settings["responseFormat"] = response_format
        payload = build_payload(
            args.model,
            {"messages": [{"role": "user", "content": "What is the weather in Sunnyvale?"}]},
            settings,
        )
        try:
            client.chat_completion(payload)
            print("    {:<38} accepted".format(label))
            return True
        except CerebrasAPIError as error:
            print("    {:<38} REJECTED {}: {}".format(
                label, error.status_code, str(error)[:130]))
            return False

    def tool_with(mutate):
        tool = copy.deepcopy(WEATHER_TOOL)
        mutate(tool["function"]["parameters"])
        return [tool]

    def strict_schema(**overrides):
        """A schema that conforms to strict mode, before overrides break it."""
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        schema.update(overrides)
        return {"type": "json", "schema": schema}

    failures = []

    def record(label, accepted, consequence):
        if not accepted:
            failures.append((label, consequence))

    # Tools, sent exactly as DSS supplies them -- no `strict`, so the permissive path.
    print("\n  tools (as the plugin sends them, no strict flag):")
    if not attempt("control, plain schema", tools=[copy.deepcopy(WEATHER_TOOL)]):
        print("\n  Control failed, so nothing below is conclusive. Check that {} "
              "supports tool calling at all.".format(args.model))
        return

    def set_id(params):
        params["$id"] = SCHEMA_ID

    def set_nested_id(params):
        params["properties"]["city"]["$id"] = SCHEMA_ID

    record("$id on tool parameters",
           attempt("$id on parameters", tools=tool_with(set_id)),
           "strip $id from each tool's parameters")
    record("$id nested in a tool property",
           attempt("$id nested in a property", tools=tool_with(set_nested_id)),
           "strip $id recursively, not just at the top level")

    # response_format, where the plugin forces strict:true on whatever DSS sent.
    print("\n  response_format (plugin forces strict:true):")
    if not attempt("control, conformant schema", response_format=strict_schema()):
        print("\n  Even a conformant strict schema was rejected -- structured outputs "
              "may be unavailable on {}.".format(args.model))
        return

    record("$id in a strict schema",
           attempt("$id present", response_format=strict_schema(**{"$id": SCHEMA_ID})),
           "strip $id before setting strict:true")

    # The documented requirement the plugin never enforces. DSS schemas that omit
    # additionalProperties are the common case, not an edge case.
    no_additional = strict_schema()
    del no_additional["schema"]["additionalProperties"]
    record("root object without additionalProperties",
           attempt("no additionalProperties (root)", response_format=no_additional),
           "inject additionalProperties:false, or stop forcing strict:true")

    nested_obj = strict_schema(properties={
        "city": {"type": "string"},
        "at": {"type": "object", "properties": {"lat": {"type": "number"}}},
    })
    record("nested object without additionalProperties",
           attempt("no additionalProperties (nested)", response_format=nested_obj),
           "inject additionalProperties:false into every object, at any depth")

    # Documented as unsupported in strict mode, and called out for qwen-3.8-27b.
    record("unsupported keyword (pattern)",
           attempt("pattern keyword", response_format=strict_schema(properties={
               "city": {"type": "string", "pattern": "^[A-Z]"}})),
           "drop unsupported keywords, or fall back to non-strict")

    # Optional fields are the common shape in generated schemas, so whether strict
    # mode demands every property in `required` decides how much normalising is safe.
    partial_required = strict_schema(
        properties={"city": {"type": "string"}, "note": {"type": "string"}},
        required=["city"],
    )
    record("required listing only some properties",
           attempt("partial required", response_format=partial_required),
           "add every property to required, or fall back to non-strict")

    no_required = strict_schema()
    del no_required["schema"]["required"]
    record("schema with no required at all",
           attempt("no required", response_format=no_required),
           "synthesise required, or fall back to non-strict")

    # Whether dropping strict rescues a non-conformant schema decides if that is a
    # viable fallback. extra_body is merged last, so it overrides response_format.
    print("\n  same non-conformant schema, strict omitted:")
    lenient = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    payload = build_payload(
        args.model,
        {"messages": [{"role": "user", "content": "Weather in Sunnyvale?"}]},
        {"temperature": 0, "maxOutputTokens": 64},
        extra_body={"response_format": json.dumps(lenient)},
    )
    try:
        client.chat_completion(payload)
        print("    {:<38} accepted".format("no strict, no additionalProperties"))
        print("      -> dropping strict is a viable fallback")
    except CerebrasAPIError as error:
        print("    {:<38} REJECTED {}: {}".format(
            "no strict, no additionalProperties", error.status_code, str(error)[:130]))
        print("      -> dropping strict does NOT rescue it; normalising is the only option")

    print("\n  verdict:")
    if not failures:
        print("    Everything tested was accepted. No normalisation needed.")
        return
    print("    {} case(s) rejected. build_payload should:".format(len(failures)))
    for label, consequence in failures:
        print("      - {}".format(consequence))
        print("          (triggered by: {})".format(label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--base-url", default="https://api.cerebras.ai/v1")
    parser.add_argument("--only",
                        choices=["models", "completion", "streaming", "tools",
                                 "structured", "schema"])
    args = parser.parse_args()

    client = _client(args)
    checks = {
        "models": check_models,
        "completion": check_completion,
        "streaming": check_streaming,
        "tools": check_tool_calling,
        "structured": check_structured_output,
        "schema": check_schema_metadata,
    }
    for name, check in checks.items():
        if args.only and args.only != name:
            continue
        check(client, args)
    print("\nAll checks passed against {}.".format(args.model))


if __name__ == "__main__":
    main()
