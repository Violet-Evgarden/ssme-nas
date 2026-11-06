# Offline concept-generation prompts

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
