# Security and release boundaries

This preview needs no authentication or network access. Do not add provider
keys, tokens, credentials, private endpoints, benchmark data, or experiment
archives. The `.gitignore` is a secondary precaution, not a secret scanner.

The release is assembled from an explicit text-file allowlist and starts with
fresh repository history. Private clients and logs are excluded rather than
copied and later deleted. Before publishing future revisions, inspect every
file and every commit for credentials and private storage details.

If you find a secret in a public repository, do not quote it in an issue.
Notify the owner privately and revoke the affected credential. Deleting a
working-tree file alone would not remove it from Git history.

The released tests and entry point do not make API requests or start training.
