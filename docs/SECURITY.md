# Security

SEME-NAS reads local benchmark files. Search with the supplied frozen concepts
requires no API key, network connection or network training. Install the
optional benchmark dependencies listed in the [execution guide](RELEASE_STATUS.md).

Download benchmark files only from trusted official sources. The NAS-Bench-201
PyTorch and NATS-Bench pickle formats deserialize Python objects and must not
be used with untrusted files. The native NAS-Bench-101 reader checks TFRecord
checksums and rejects malformed records.

Do not commit credentials, access tokens, private endpoints, local benchmark
tables or generated run outputs. Inspect changes before publishing; ignore
rules are a safeguard, not a substitute for review.

If a credential is exposed, contact the owner privately and revoke it. Do not
paste it into a public issue; deleting a file alone does not erase Git history.
