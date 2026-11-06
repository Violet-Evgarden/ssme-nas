# Copyright and third-party notices

Copyright (c) 2026 SEME-NAS contributors. All rights reserved, except for the
third-party material identified below.

The repository publishes the complete SEME-NAS search implementation and its
frozen concept and initialization artifacts. No new open-source license is
granted for the authors' original code by this publication. Rights available
through the hosting platform's applicable terms are unaffected. Contact the
repository owner for additional permissions.

## NAS-Bench-101 graph hashing

The graph-hashing portion of `official()` in
`ssme_nas/core/nb101.py` is adapted from `hash_module()` in
[Google NASBench, nasbench/lib/graph_util.py](https://github.com/google-research/nasbench/blob/master/nasbench/lib/graph_util.py).

Copyright 2019 The Google Research Authors.

This portion is licensed under the Apache License, Version 2.0. A copy is
included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

The adaptation uses Python tuples and standard-library operations in place of
NumPy arrays and receives the effective graph from the native genotype
adapter. It preserves the original iterative MD5 graph-hashing procedure.
The rest of the SEME-NAS implementation is not covered by this third-party
license.

Unless required by applicable law or agreed to in writing, this third-party
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
OF ANY KIND, either express or implied. See the Apache License for its
permissions and limitations.

## Benchmark resources

Benchmark packages and benchmark tables are obtained separately from their
official providers and remain subject to their respective licenses and access
terms.
