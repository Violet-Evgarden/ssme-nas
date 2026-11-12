"""Stage-restricted benchmark evaluation with one sampled validation repeat.

Supported inputs are the official NAS-Bench-101 TFRecord, NAS-Bench-201 .pth,
NATS-Bench topology .pickle.pbz2, and an exported JSON table. Benchmark packages
are optional and imported only when their corresponding input is selected.
The default NB101 reader uses the official TFRecord and protobuf layouts without
TensorFlow, verifies every TFRecord checksum, and decodes only requested metrics.
The NB201 APIs deserialize complete tables; this wrapper reads and returns only
the requested metric. Search code receives no test metric. Test reads are
enabled only after every configured search run has exhausted its query budget
and its validation winner has been frozen.

API references:
https://github.com/google-research/nasbench/blob/master/nasbench/api.py
https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_metrics.proto
https://github.com/D-X-Y/NAS-Bench-201/blob/master/nas_201_api/api_utils.py
https://github.com/D-X-Y/NATS-Bench/blob/main/nats_bench/api_topology.py
"""
import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import re
import statistics
import struct

from ._utils import digest

try:
    from google_crc32c import value as _accelerated_crc32c
except ImportError:
    _accelerated_crc32c = None


class BenchmarkDataError(ValueError):
    """Required benchmark data are missing, malformed, or non-finite."""


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _protocol(config):
    name, dataset = config["space"], config["dataset"]
    if name == "nb101" and dataset == "cifar10":
        expected = {
            "validation": {"budget": 108, "metric": "final_validation_accuracy"},
            "test": {"budget": 108, "metric": "final_test_accuracy"},
        }
    elif name == "nb201" and dataset in ("cifar10", "cifar100", "ImageNet16-120"):
        expected = {
            "validation": {"dataset": "cifar10-valid" if dataset == "cifar10" else dataset,
                           "hp": 200, "epoch_index": 199, "metric": "x-valid"},
            "test": {"dataset": dataset, "hp": 200, "epoch_index": 199,
                     "metric": "ori-test" if dataset == "cifar10" else "x-test"},
        }
    else:
        raise ValueError(f"Unsupported benchmark/dataset: {name}/{dataset}")
    for stage, fields in expected.items():
        fields["repeat_aggregation"] = ("random_one_of_three" if stage == "validation"
                                        else "all_available_mean")
        for key, value in fields.items():
            if config.get(stage, {}).get(key) != value:
                raise ValueError(f"Unsupported {stage}.{key}; expected {value!r}.")
    return expected


def _repeats(rows):
    result = []
    for row in rows:
        repeat = _integer(row["repeat_id"], "repeat_id")
        if isinstance(row["value"], bool):
            raise BenchmarkDataError("Accuracy must be numeric, not boolean.")
        value = float(row["value"])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise BenchmarkDataError("Accuracy must be finite and in [0, 100].")
        result.append({"repeat_id": repeat, "value": value})
    result.sort(key=lambda row: row["repeat_id"])
    if not result or len({row["repeat_id"] for row in result}) != len(result):
        raise BenchmarkDataError("Expected nonempty, uniquely identified official repeats.")
    return result


class BenchmarkEvaluator:
    """One evaluator is shared by all configured runs, before winner-only Test."""

    def __init__(self, config, space, backend):
        self.config = deepcopy(config)
        self.protocol = _protocol(config)
        self.space = space
        self._backend = backend
        self._runs = _integer(config["search_runs"], "search_runs")
        self._budget = _integer(config["queries_per_run"], "queries_per_run")
        if self._runs < 1 or self._budget < 1:
            raise ValueError("Run count and query budget must be positive.")
        self._queries = {pool: {} for pool in range(self._runs)}
        self._search_seeds = {}
        self._cache = {}
        self._winners = None
        self._closed = False

    def _identity(self, ops, intent):
        if self._closed:
            raise RuntimeError("Evaluator is closed.")
        identity = self.space.canonical(tuple(ops))
        if intent.get("architecture_id") != identity:
            raise ValueError("Evaluation intent and genotype identity disagree.")
        pool = _integer(intent.get("pool"), "pool")
        if pool not in self._queries:
            raise ValueError("Pool lies outside the configured runs.")
        return identity, pool

    def _measurement(self, identity, stage, pool=None, seed=None):
        # Validation draws belong to a run, not the globally shared architecture.
        cache_key = (stage, pool, identity) if stage == "validation" else (stage, identity)
        if cache_key not in self._cache:
            try:
                selected = None
                if stage == "validation":
                    available = sorted(_integer(value, "repeat_id") for value in
                                       self._backend.repeat_ids(identity, stage, self.protocol[stage]))
                    if len(available) != 3 or len(set(available)) != 3:
                        raise BenchmarkDataError("Validation requires exactly three distinct official repeats.")
                    # Separate namespace: mutation RNG and query ordering cannot
                    # consume or change the validation draw for this architecture.
                    draw_seed = digest(["validation-repeat-v1", self.config["space"],
                                        self.config["dataset"], seed, identity])
                    selected = random.Random(draw_seed).choice(available)
                rows = _repeats(self._backend.read(identity, stage, self.protocol[stage], selected))
                if selected is not None and [row["repeat_id"] for row in rows] != [selected]:
                    raise BenchmarkDataError("Backend must return only the selected validation repeat.")
            except (KeyError, IndexError, TypeError) as exc:
                raise BenchmarkDataError(f"Missing {stage} repeats for {identity}.") from exc
            self._cache[cache_key] = {
                "key": {"architecture_id": identity, "space": self.config["space"],
                        "dataset": self.protocol[stage].get("dataset", self.config["dataset"]),
                        **({"budget": 108, "metric": self.protocol[stage]["metric"]}
                           if self.config["space"] == "nb101" else
                           {"hp": 200, "epoch_index": 199, "split": self.protocol[stage]["metric"]}),
                        "aggregation": ("random_one_of_three" if stage == "validation"
                                        else "all_actual_official_repeats_mean")},
                "architecture_id": identity,
                "stage": stage,
                "value": (rows[0]["value"] if stage == "validation"
                          else statistics.mean(row["value"] for row in rows)),
                "repeats": rows,
                "read_repeats": rows,
                "repeat_ids": [row["repeat_id"] for row in rows],
                "repeat_count": len(rows),
                "protocol": deepcopy(self.protocol[stage]),
            }
            if stage == "validation":
                self._cache[cache_key].update(selected_repeat_id=selected,
                                              available_repeat_ids=available,
                                              search_seed=seed)
        return deepcopy(self._cache[cache_key])

    def validation(self, ops, intent):
        identity, pool = self._identity(ops, intent)
        if self._winners is not None:
            raise RuntimeError("Validation is closed after the winner seal.")
        seed = _integer(intent.get("search_seed"), "search_seed")
        if seed < 0:
            raise ValueError("search_seed must be nonnegative.")
        if pool in self._search_seeds and self._search_seeds[pool] != seed:
            raise ValueError("The search seed is immutable within a run.")
        index = _integer(intent.get("query_index"), "query_index")
        queries = self._queries[pool]
        if not 0 <= index < self._budget:
            raise ValueError("Validation query budget exhausted.")
        if index in queries:
            if queries[index] != identity:
                raise ValueError("A recorded query intent cannot be changed.")
        elif index != len(queries) or identity in queries.values():
            raise ValueError("Queries must be sequential and distinct within each run.")
        measurement = self._measurement(identity, "validation", pool, seed)
        self._search_seeds[pool] = seed
        queries[index] = identity
        return measurement

    def seal_winners(self, winners):
        """Freeze every run's unrounded validation argmax before any Test read."""
        if self._closed:
            raise RuntimeError("Evaluator is closed.")
        winners = list(winners)
        if len(winners) != self._runs:
            raise ValueError("Exactly one winner is required for every configured run.")
        selected = {}
        for winner in winners:
            pool = _integer(winner.get("pool"), "pool")
            if pool not in self._queries or pool in selected:
                raise ValueError("Winner pools must cover each configured run exactly once.")
            queried = self._queries[pool]
            if len(queried) != self._budget:
                raise ValueError("All runs must finish their validation budgets before Test.")
            best = min(queried.values(), key=lambda identity: (
                -self._cache["validation", pool, identity]["value"], identity))
            if winner.get("architecture_id") != best:
                raise ValueError("Winner is not the validation argmax with canonical tie-breaking.")
            selected[pool] = best
        if self._winners is not None and self._winners != selected:
            raise ValueError("Winner seal is immutable.")
        self._winners = selected

    def test(self, ops, intent):
        identity, pool = self._identity(ops, intent)
        if self._winners is None or self._winners.get(pool) != identity:
            raise PermissionError("Test is available only for globally sealed winners.")
        return self._measurement(identity, "test")

    def architecture_ids(self):
        """Official identities only; no performance metric is read."""
        if self._closed:
            raise RuntimeError("Evaluator is closed.")
        return tuple(sorted(self._backend.identities()))

    def structures(self):
        """Yield label-free metadata for constructing random initial genotypes."""
        for identity in self.architecture_ids():
            yield deepcopy(self._backend.structure(identity))

    def close(self):
        self._backend.close()
        self._closed = True


class _JsonTable:
    def __init__(self, path, config):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("format") != "seme-nas-benchmark-v1":
            raise BenchmarkDataError("Unknown JSON benchmark format.")
        if any(data.get(key) != config[key] for key in ("space", "dataset")):
            raise BenchmarkDataError("JSON benchmark space/dataset mismatch.")
        for stage, protocol in _protocol(config).items():
            if any(data.get(stage, {}).get(key) != value for key, value in protocol.items()):
                raise BenchmarkDataError(f"JSON benchmark {stage} protocol mismatch.")
        self._entries = data["architectures"]
        if not isinstance(self._entries, dict) or not self._entries:
            raise BenchmarkDataError("JSON benchmark must contain architecture records.")

    def identities(self):
        return self._entries.keys()

    def structure(self, identity):
        entry = self._entries[identity]
        return {"architecture_id": identity, **{
            key: entry[key] for key in ("ops", "matrix") if key in entry}}

    def repeat_ids(self, identity, stage, protocol):
        return [row["repeat_id"] for row in self._entries[identity][stage]]

    def read(self, identity, stage, protocol, selected=None):
        return [row for row in self._entries[identity][stage]
                if selected is None or row["repeat_id"] == selected]

    def close(self):
        pass


def _crc_table():
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)
    return tuple(table)


_CRC32C_TABLE = _crc_table()


def _crc32c(data):
    if _accelerated_crc32c is not None:
        return _accelerated_crc32c(data)
    value = 0xFFFFFFFF
    for byte in data:
        value = _CRC32C_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def _masked_crc32c(data):
    value = _crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _tfrecord(handle):
    """Read one checksum-verified TFRecord; None marks a clean record boundary EOF."""
    offset = handle.tell()
    length_bytes = handle.read(8)
    if not length_bytes:
        return None
    if len(length_bytes) != 8:
        raise BenchmarkDataError(f"Truncated TFRecord length at offset {offset}.")
    length_crc = handle.read(4)
    if len(length_crc) != 4 or struct.unpack("<I", length_crc)[0] != _masked_crc32c(length_bytes):
        raise BenchmarkDataError(f"TFRecord length checksum mismatch at offset {offset}.")
    length = struct.unpack("<Q", length_bytes)[0]
    # NAS-Bench rows are small; reject corrupted lengths before allocating RAM.
    if not 0 < length <= 16 * 1024 * 1024:
        raise BenchmarkDataError(f"Invalid NAS-Bench TFRecord size at offset {offset}.")
    payload, checksum = handle.read(length), handle.read(4)
    if len(payload) != length or len(checksum) != 4:
        raise BenchmarkDataError(f"Truncated TFRecord payload at offset {offset}.")
    if struct.unpack("<I", checksum)[0] != _masked_crc32c(payload):
        raise BenchmarkDataError(f"TFRecord data checksum mismatch at offset {offset}.")
    return payload


def _varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise BenchmarkDataError("Truncated protobuf varint.")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
    raise BenchmarkDataError("Invalid protobuf varint.")


def _wire_fields(data):
    """Iterate raw protobuf fields; unrelated floating-point metrics stay undecoded."""
    position = 0
    while position < len(data):
        tag, position = _varint(data, position)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise BenchmarkDataError("Invalid protobuf field number.")
        if wire == 0:
            value, position = _varint(data, position)
        else:
            if wire == 1:
                length = 8
            elif wire == 2:
                length, position = _varint(data, position)
            elif wire == 5:
                length = 4
            else:
                raise BenchmarkDataError(f"Unsupported protobuf wire type {wire}.")
            end = position + length
            if end > len(data):
                raise BenchmarkDataError("Truncated protobuf field.")
            value, position = data[position:end], end
        yield number, wire, value


def _final_nb101_metric(encoded, stage):
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise BenchmarkDataError("Malformed base64 NAS-Bench metrics.") from exc
    evaluations = [value for number, wire, value in _wire_fields(raw) if number == 1 and wire == 2]
    if len(evaluations) < 3:
        raise BenchmarkDataError("The official final evaluation checkpoint is missing.")
    # Official api.py uses evaluation_data[2]; final accuracy fields are doubles
    # 4 (Validation) and 5 (Test) in model_metrics.proto.
    target = 4 if stage == "validation" else 5
    selected = [value for number, wire, value in _wire_fields(evaluations[2])
                if number == target and wire == 1]
    if len(selected) != 1:
        raise BenchmarkDataError(f"Missing or duplicated final {stage} metric.")
    return struct.unpack("<d", selected[0])[0] * 100.0


class _NasBench101TFRecord:
    """Label-free offset index; selected metrics are decoded only when queried."""

    def __init__(self, path):
        self._path = path
        self._offsets = {}
        self._handle = path.open("rb")
        try:
            while True:
                offset = self._handle.tell()
                payload = _tfrecord(self._handle)
                if payload is None:
                    break
                identity, epochs, adjacency, operations, encoded = self._row(payload)
                if epochs == 108:
                    self._offsets.setdefault(identity, []).append(offset)
            if not self._offsets:
                raise BenchmarkDataError("The TFRecord contains no 108-epoch records.")
        except Exception:
            self._handle.close()
            raise

    @staticmethod
    def _row(payload):
        try:
            row = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchmarkDataError("Malformed JSON in NAS-Bench TFRecord.") from exc
        if (not isinstance(row, list) or len(row) != 5 or not isinstance(row[0], str)
                or isinstance(row[1], bool) or not isinstance(row[1], int)
                or not all(isinstance(item, str) for item in row[2:])):
            raise BenchmarkDataError("Malformed NAS-Bench TFRecord row.")
        return row

    def _read_offset(self, offset):
        self._handle.seek(offset)
        payload = _tfrecord(self._handle)
        if payload is None:
            raise BenchmarkDataError("Indexed TFRecord is no longer present.")
        return self._row(payload)

    def identities(self):
        return self._offsets.keys()

    def structure(self, identity):
        row = self._read_offset(self._offsets[identity][0])
        nodes = row[3].split(",")
        n = len(nodes)
        if len(row[2]) != n * n or any(value not in "01" for value in row[2]):
            raise BenchmarkDataError("Invalid NAS-Bench adjacency matrix.")
        return {"architecture_id": identity, "ops": nodes,
                "matrix": [[int(value) for value in row[2][i*n:(i+1)*n]] for i in range(n)]}

    def repeat_ids(self, identity, stage, protocol):
        return range(len(self._offsets[identity]))

    def read(self, identity, stage, protocol, selected=None):
        result = []
        for repeat, offset in enumerate(self._offsets[identity]):
            if selected is not None and repeat != selected:
                continue
            row = self._read_offset(offset)
            if row[0] != identity or row[1] != 108:
                raise BenchmarkDataError("TFRecord no longer agrees with its offset index.")
            result.append({"repeat_id": repeat, "value": _final_nb101_metric(row[4], stage)})
        return result

    def close(self):
        self._handle.close()


class _NasBench101:
    def __init__(self, path):
        try:
            from nasbench import api
        except ImportError as exc:
            raise ImportError("Install the official google-research/nasbench package and its "
                              "TensorFlow-compatible environment to read a TFRecord.") from exc
        self._api = api.NASBench(str(path))

    def identities(self):
        return self._api.fixed_statistics.keys()

    def structure(self, identity):
        record = self._api.fixed_statistics[identity]
        matrix = record["module_adjacency"]
        return {"architecture_id": identity,
                "matrix": matrix.tolist() if hasattr(matrix, "tolist") else matrix,
                "ops": list(record["module_operations"])}

    def repeat_ids(self, identity, stage, protocol):
        return range(len(self._api.computed_statistics[identity][protocol["budget"]]))

    def read(self, identity, stage, protocol, selected=None):
        # Do not call query/get_metrics_from_hash: those return test fields too.
        entries = self._api.computed_statistics[identity][protocol["budget"]]
        field = protocol["metric"]
        return [{"repeat_id": index, "value": float(record[field]) * 100.0}
                for index, record in enumerate(entries) if selected is None or index == selected]

    def close(self):
        self._api = None


class _NasBench201:
    def __init__(self, path):
        if path.suffix.lower() == ".pth":
            try:
                import torch
                from nas_201_api import NASBench201API
            except ImportError as exc:
                raise ImportError("Install nas-bench-201 and torch to read the official .pth table.") from exc
            # Official benchmark files are pickled data: load only trusted copies.
            # Explicit weights_only=False supports PyTorch >=2.6 as well as 2.x.
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            self._api = NASBench201API(data, verbose=False)
        else:
            try:
                from nats_bench import create
            except ImportError as exc:
                raise ImportError("Install nats-bench to read the NATS topology .pickle.pbz2 table.") from exc
            self._api = create(str(path), "tss", fast_mode=False, verbose=False)

    def identities(self):
        return self._api.meta_archs

    def structure(self, identity):
        return {"architecture_id": identity, "ops": re.findall(r"([^|+~]+)~\d+", identity)}

    def _meta(self, identity, protocol):
        index = self._api.query_index_by_arch(identity)
        if index < 0:
            raise BenchmarkDataError("Architecture is absent from the official NB201 table.")
        return self._api.arch2infos_dict[index][str(protocol["hp"])]

    def repeat_ids(self, identity, stage, protocol):
        return self._meta(identity, protocol).get_dataset_seeds(protocol["dataset"])

    def read(self, identity, stage, protocol, selected=None):
        # Full-table mode: read the selected dataset/metric, not get_more_info(),
        # which bundles validation and test results in a single response.
        meta = self._meta(identity, protocol)
        dataset = protocol["dataset"]
        repeats = meta.get_dataset_seeds(dataset)
        if len(repeats) != len(set(repeats)):
            raise BenchmarkDataError("Official repeat IDs are duplicated.")
        return [{"repeat_id": int(seed), "value": meta.all_results[dataset, seed].get_eval(
            protocol["metric"], protocol["epoch_index"])["accuracy"]} for seed in repeats
                if selected is None or seed == selected]

    def close(self):
        self._api = None


def create_evaluator(config, benchmark_path, space):
    """Create a sealed evaluator; validation samples one of three official repeats.

    JSON values must already be percentages. Each exported record has an ops
    genotype (or NB101 matrix + node ops) and validation/test lists of
    {repeat_id: int, value: float}; the file declares space, dataset, and both
    evaluation protocols. Missing repeats/metrics raise instead of being skipped.
    """
    _protocol(config)
    path = Path(benchmark_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file does not exist: {path}")
    if path.suffix.lower() == ".json":
        backend = _JsonTable(path, config)
    elif config["space"] == "nb101":
        if path.suffix.lower() != ".tfrecord":
            raise ValueError("NAS-Bench-101 requires an official .tfrecord or exported .json.")
        reader = config.get("nb101_reader", "tfrecord")
        if reader not in ("tfrecord", "nasbench_api"):
            raise ValueError("nb101_reader must be 'tfrecord' or 'nasbench_api'.")
        backend = _NasBench101(path) if reader == "nasbench_api" else _NasBench101TFRecord(path)
    elif path.suffix.lower() in (".pth", ".pbz2"):
        backend = _NasBench201(path)
    else:
        raise ValueError("NAS-Bench-201 requires official .pth, .pickle.pbz2, or exported .json.")
    try:
        return BenchmarkEvaluator(config, space, backend)
    except Exception:
        backend.close()
        raise
