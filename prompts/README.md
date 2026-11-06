# Offline concept generation

The `*_flash_request.json` files are the complete request bodies used for the
NAS-Bench-101 and NAS-Bench-201 concept constructions. They are recorded inputs,
not rewritten templates or generated outputs. Their contents, including the
system and user message strings and generation settings, are unchanged.

| Request | Structural fields | Unlabelled examples | Requested bundles |
|---|---:|---:|---:|
| `nb101_flash_request.json` | 22 | 16 | 8–12 |
| `nb201_flash_request.json` | 22 | 16 | 12 |

Each bundle uses one to three existing field names. Bundles jointly cover the
field catalog, with no duplicate feature sets. The user message is a JSON string
containing `grammar`, `feature_catalog`, `examples` and construction instructions.
The `performance` entries are the literal `NOT_QUERIED_OR_PROVIDED` marker,
not numerical observations.

The `*_response.schema.json` files document the expected response structure.
They were added for readers and were not sent as provider-side structured-output
schemas: the recorded requests use `response_format: {"type": "json_object"}`.
Full field coverage and order-independent bundle uniqueness are additional
cross-bundle validation checks.

The request model identifier and decoding settings are preserved as recorded.
These files do not contain credentials or perform API calls.

## Generate and save concepts

1. Choose [`nb101_flash_request.json`](nb101_flash_request.json) or
   [`nb201_flash_request.json`](nb201_flash_request.json). Submit its complete
   request body through your configured model provider's client. Preserve the
   messages and generation settings; keep credentials outside the JSON file.
2. Extract the assistant's JSON content from the response. It must be an object
   with a `concepts` array, not the provider's surrounding response envelope.
   Save this object locally as `concepts.json`.
3. Check JSON syntax with `python -m json.tool concepts.json`. Validate the object
   against the corresponding [`NB101`](nb101_response.schema.json) or
   [`NB201`](nb201_response.schema.json) response schema, then check the
   cross-bundle constraints below.
4. Retain the validated `concepts.json`, original request and complete provider
   response together. Read `concepts.json` and pass its `concepts` array to the
   controller's `concepts` argument. Reuse the fixed array during search; no
   further LLM call is needed for the mutation loop.

## Validation checklist

- NB101 has 8–12 bundles; NB201 has 12.
- Each bundle contains only `features` and `meaning`. Its `features` array has
  one to three distinct names from the selected request's `feature_catalog`;
  `meaning` is a nonempty string.
- The union of all bundle features equals the complete field catalog.
- No two bundles contain the same feature set, irrespective of feature order.

The schemas cover individual bundle structure and vocabulary. Full catalog
coverage and order-independent bundle uniqueness must also be checked across
the complete `concepts` array. The syntax command alone does not check these
constraints.
