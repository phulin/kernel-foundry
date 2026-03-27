import numpy as np
import pytest
from kernel_foundry.archive.island_archive import IslandArchive
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

    def test_equal_fitness_higher_speedup_replaces_incumbent(self):
        a = MAPElitesArchive()
        r_low = make_record(1, 0, 0, speedup=5.33, kid="low")
        r_high = make_record(1, 0, 0, speedup=5.66, kid="high")
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

    def test_get_active_variant_empty_returns_none(self):
        pa = PromptArchive(capacity=5)
        rng = np.random.default_rng(0)
        assert pa.get_active_variant(rng) is None

    def test_get_active_variant_explores_with_epsilon(self):
        rng = np.random.default_rng(42)
        pa = PromptArchive(capacity=10)
        v1 = pa.insert(DEFAULT_SECTIONS, generation=0)
        v2 = pa.insert(DEFAULT_SECTIONS, generation=1)
        pa.update_fitness(v1.variant_id, 0.9)  # v1 is best
        # With epsilon=1.0 (always explore), should sometimes return v2
        seen = set()
        for _ in range(50):
            v = pa.get_active_variant(rng, epsilon=1.0)
            seen.add(v.variant_id)
        assert v2.variant_id in seen

    def test_get_active_variant_exploits_with_zero_epsilon(self):
        rng = np.random.default_rng(42)
        pa = PromptArchive(capacity=10)
        v1 = pa.insert(DEFAULT_SECTIONS, generation=0)
        pa.insert(DEFAULT_SECTIONS, generation=1)
        pa.update_fitness(v1.variant_id, 0.9)
        # With epsilon=0 (always exploit), should always return best
        for _ in range(20):
            v = pa.get_active_variant(rng, epsilon=0.0)
            assert v.variant_id == v1.variant_id


# ──────────────────────────────────────────── IslandArchive

class TestIslandArchive:
    def test_insert_into_current_island(self):
        ia = IslandArchive(n_islands=2, bins=4)
        from tests.conftest import make_record
        assert ia.insert(make_record(1, 0, 0, speedup=1.5)) is True
        assert ia.size() == 1

    def test_islands_are_isolated_before_migration(self):
        from tests.conftest import make_record
        ia = IslandArchive(n_islands=2, migration_freq=10, bins=4)
        ia.insert(make_record(1, 0, 0, speedup=2.0, kid="i0"))
        # Switch to island 1 manually — it should be empty
        ia._current = 1
        assert ia.size() == 0

    def test_migration_broadcasts_elites_to_all_islands(self):
        from tests.conftest import make_record
        ia = IslandArchive(n_islands=2, migration_freq=5, bins=4)
        ia.insert(make_record(1, 0, 0, speedup=2.0, kid="i0_elite"))
        ia._current = 1
        ia.insert(make_record(2, 1, 0, speedup=1.5, kid="i1_elite"))
        ia._current = 0
        ia._migrate()
        # Island 0 should now have i1's elite
        ia._current = 0
        assert ia.get_elite(BehavioralCoords(2, 1, 0)) is not None
        # Island 1 should now have i0's elite
        ia._current = 1
        assert ia.get_elite(BehavioralCoords(1, 0, 0)) is not None

    def test_advance_generation_triggers_at_migration_freq(self):
        from tests.conftest import make_record
        ia = IslandArchive(n_islands=2, migration_freq=5, bins=4)
        ia.insert(make_record(1, 0, 0, speedup=2.0))
        ia.advance_generation(1)
        assert ia._current == 1
        ia.advance_generation(2)
        assert ia._current == 0
        ia.advance_generation(5)  # still rotates; also triggers migration
        assert ia._current == 1

    def test_set_current_island_wraps_index(self):
        ia = IslandArchive(n_islands=3, bins=4)
        ia.set_current_island(4)
        assert ia.current_island == 1

    def test_get_best_overall_across_islands(self):
        from tests.conftest import make_record
        ia = IslandArchive(n_islands=2, bins=4)
        ia.insert(make_record(0, 0, 0, speedup=1.5, kid="low"))
        ia._current = 1
        ia.insert(make_record(1, 1, 0, speedup=3.0, kid="high"))
        ia._current = 0
        best = ia.get_best_overall()
        assert best.kernel_id == "high"

    def test_get_all_elites_deduplicates_by_fitness(self):
        from tests.conftest import make_record
        ia = IslandArchive(n_islands=2, bins=4)
        ia.insert(make_record(1, 0, 0, speedup=1.5, kid="low"))
        ia._current = 1
        ia.insert(make_record(1, 0, 0, speedup=2.5, kid="high"))
        ia._current = 0
        elites = ia.get_all_elites()
        kids = {r.kernel_id for r in elites}
        assert "high" in kids
        assert "low" not in kids

    def test_best_overall_empty_returns_none(self):
        ia = IslandArchive(n_islands=3, bins=4)
        assert ia.get_best_overall() is None
