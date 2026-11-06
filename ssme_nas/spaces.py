"""Native, label-free NAS-Bench-101 and NAS-Bench-201 adapters."""
from .core import nb101, nb201


class NativeSpace:
    """Bind graph operations to one space without process-global monkeypatches."""

    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.CATALOG = module.CATALOG

    def canonical(self, genotype):
        return self.module.canonical(tuple(genotype))

    def validate(self, genotype):
        try:
            self.canonical(genotype)
        except (AssertionError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {self.name} genotype") from exc

    def validate_edit(self, parent, child):
        self.validate(parent)
        self.validate(child)
        if self.canonical(parent) == self.canonical(child):
            raise ValueError("An edit must change the canonical architecture identity")
        if not 1 <= len(self.action(parent, child)) <= 2:
            raise ValueError("A native mutation changes one or two aligned slots")

    def action(self, parent, child):
        return self.module.action(tuple(parent), tuple(child))

    def action_type(self, parent, child):
        return self.module.action_type(tuple(parent), tuple(child))

    def features(self, parent, child):
        return self.module.features(tuple(parent), tuple(child))

    def candidates(self, parent, excluded):
        return self.module.candidates(tuple(parent), excluded)

    def validate_measurement(self, record):
        """Check the same validation endpoint used by the archived main runs."""
        key = record['validation']['key']
        if key['architecture_id'] != record['architecture_id']:
            raise ValueError("Measurement identity does not match its architecture")
        if self.name == 'nb101':
            if key['metric'] != 'final_validation_accuracy':
                raise ValueError("NAS-Bench-101 search requires final validation accuracy")
        elif not (key['split'] == 'x-valid' and key['hp'] == 200 and key['epoch_index'] == 199):
            raise ValueError("NAS-Bench-201 search requires the 200-epoch final x-valid endpoint")


_SPACES = {'nb101': NativeSpace('nb101', nb101), 'nb201': NativeSpace('nb201', nb201)}


def get_space(name):
    """Return a native adapter; graph validity never reads performance labels."""
    if isinstance(name, NativeSpace):
        return name
    try:
        return _SPACES[str(name).lower()]
    except KeyError as exc:
        raise ValueError("Expected search space 'nb101' or 'nb201'") from exc
