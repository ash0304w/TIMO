import pytest

from unifsl_rl.core.conflict_checker import check_slot_conflicts


class A:
    id = "a"
    owns = ["alpha"]


class B:
    id = "b"
    owns = ["alpha"]


def test_slot_conflict_raise():
    with pytest.raises(RuntimeError):
        check_slot_conflicts([A(), B()])
