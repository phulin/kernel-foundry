import pytest
from kernel_foundry.archive.map_elites import MAPElitesArchive
from kernel_foundry.archive.prompt_archive import PromptArchive
from kernel_foundry.types import BehavioralCoords
from kernel_foundry.prompt.evolvable_sections import DEFAULT_SECTIONS
from tests.conftest import make_record


# ──────────────────────────────────────────── MAPElitesArchive

class TestMAPElitesArchive:
    def test_empty_on_init(self):
        a = MAPElitesArchive(bins=4)
        assert a.size() == 0
        assert a.get_best_overall() is None
        assert a.get_occupied_cells() == []

    def test_insert_into_empty_cell(self):
        a = MAPElitesArchive()
        r = make_record(1, 0, 0, speedup=1.5)
        assert a.insert(r) is True
        assert a.size() == 1

    def test_no_improvement_returns_false(self):
        a = MAPElitesArchive()
        r = make_record(1, 0, 0, speedup=1.5)
        a.insert(r)
        assert a.insert(r) is False  # same fitness
        assert a.size() == 1

    def test_higher_fitness_replaces_incumbent(self):
        a = MAPElitesArchive()
        r_low = make_record(1, 0, 0, speedup=1.5, kid="low")
        r_high = make_record(1, 0, 0, speedup=2.5, kid="high")
        a.insert(r_low)
        assert a.insert(r_high) is True
        assert a.get_elite(BehavioralCoords(1, 0, 0)).kernel_id == "high"

    def test_lower_fitness_does_not_replace(self):
        a = MAPElitesArchive()
        r_high = make_record(1, 0, 0, speedup=2.5, kid="high")
        r_low = make_record(1, 0, 0, speedup=1.0, kid="low")
        a.insert(r_high)
        assert a.insert(r_low) is False
        assert a.get_elite(BehavioralCoords(1, 0, 0)).kernel_id == "high"

    def test_different_cells_coexist(self):
        a = MAPElitesArchive()
        a.insert(make_record(0, 0, 0, speedup=1.0))
        a.insert(make_record(1, 1, 1, speedup=2.0))
        a.insert(make_record(3, 3, 3, speedup=3.0))
        assert a.size() == 3

    def test_get_best_uses_speedup_as_tiebreaker(self):
        # Both exceed 2x target → same fitness (1.0); best should be highest speedup
        a = MAPElitesArchive()
        a.insert(make_record(1, 0, 0, speedup=2.5, kid="r_a"))
        a.insert(make_record(2, 1, 0, speedup=3.0, kid="r_b"))
        best = a.get_best_overall()
        assert best.kernel_id == "r_b"

    def test_get_fitness_empty_cell_is_zero(self):
        a = MAPElitesArchive()
        assert a.get_fitness(BehavioralCoords(0, 0, 0)) == 0.0

    def test_get_fitness_occupied_cell(self):
        a = MAPElitesArchive()
        a.insert(make_record(2, 0, 0, speedup=1.5))
        f = a.get_fitness(BehavioralCoords(2, 0, 0))
        assert f > 0.0

    def test_empty_cells_count(self):
        a = MAPElitesArchive(bins=4)
        assert len(a.get_empty_cells()) == 64
        a.insert(make_record(0, 0, 0))
        assert len(a.get_empty_cells()) == 63

    def test_compile_failed_does_not_evict_correct(self):
        a = MAPElitesArchive()
        r_correct = make_record(1, 1, 1, speedup=1.5)
        r_bad = make_record(1, 1, 1, compiled=False)
        a.insert(r_correct)
        assert a.insert(r_bad) is False  # fitness 0.0 < 0.875
        assert a.get_elite(BehavioralCoords(1, 1, 1)).eval_result.correct is True

    def test_to_dict_roundtrip_keys(self):
        a = MAPElitesArchive()
        a.insert(make_record(1, 2, 3, speedup=1.8))
        d = a.to_dict()
        assert "1,2,3" in d["cells"]
        assert d["cells"]["1,2,3"]["speedup"] == pytest.approx(1.8)


# ──────────────────────────────────────────── PromptArchive

class TestPromptArchive:
    def test_insert_and_retrieve(self):
        pa = PromptArchive(capacity=5)
        v = pa.insert(DEFAULT_SECTIONS, generation=0)
        assert pa.size() == 1
        assert pa.get_best_variant().variant_id == v.variant_id

    def test_update_fitness(self):
        pa = PromptArchive(capacity=5)
        v = pa.insert(DEFAULT_SECTIONS, generation=0)
        pa.update_fitness(v.variant_id, 0.9)
        assert pa.get_best_variant().best_fitness == 0.9

    def test_best_variant_tracks_highest_fitness(self):
        pa = PromptArchive(capacity=5)
        v1 = pa.insert(DEFAULT_SECTIONS, generation=0)
        v2 = pa.insert(DEFAULT_SECTIONS, generation=1)
        pa.update_fitness(v1.variant_id, 0.7)
        pa.update_fitness(v2.variant_id, 0.9)
        assert pa.get_best_variant().variant_id == v2.variant_id

    def test_evicts_lowest_fitness_when_full(self):
        pa = PromptArchive(capacity=3)
        v1 = pa.insert(DEFAULT_SECTIONS, generation=0)
        v2 = pa.insert(DEFAULT_SECTIONS, generation=1)
        v3 = pa.insert(DEFAULT_SECTIONS, generation=2)
        pa.update_fitness(v2.variant_id, 0.9)
        # Inserting a 4th should evict v1 or v3 (both fitness=0)
        v4 = pa.insert(DEFAULT_SECTIONS, generation=3)
        assert pa.size() == 3
        ids = {v.variant_id for v in pa.all_variants()}
        assert v2.variant_id in ids       # high-fitness variant preserved
        assert v4.variant_id in ids       # new variant inserted
        assert v1.variant_id not in ids or v3.variant_id not in ids  # one was evicted

    def test_update_fitness_only_increases(self):
        pa = PromptArchive(capacity=5)
        v = pa.insert(DEFAULT_SECTIONS, generation=0)
        pa.update_fitness(v.variant_id, 0.8)
        pa.update_fitness(v.variant_id, 0.3)  # lower — should not decrease
        assert pa.get_best_variant().best_fitness == 0.8

    def test_get_best_variant_empty_returns_none(self):
        pa = PromptArchive(capacity=5)
        assert pa.get_best_variant() is None
