"""Embedding-informed card prior: frozen projection and augmented feature space."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gigaevo.memory.cards import Card
from gigaevo.memory_v2.embedding import (
    CardEmbedder,
    CardEmbeddingReducer,
    FrozenProjection,
)
from gigaevo.memory_v2.features import (
    EmbeddingPriorConfig,
    FeatureConfig,
    HierarchicalFeatureMap,
)
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    OutcomeMeasurement,
)
from gigaevo.memory_v2.posterior import (
    HierarchicalTerminalUtilityPosterior,
    TerminalUtilityPosteriorConfig,
)

BEHAVIOR_KEYS = ("hop_depth", "passages_fetched", "instr_chars")


def _space(embedding_prior, cards, embeddings=None):
    config = FeatureConfig(behavior_keys=BEHAVIOR_KEYS, embedding_prior=embedding_prior)
    return HierarchicalFeatureMap(config=config).space(cards, embeddings=embeddings)


class TestFrozenProjection:
    """A seeded Johnson-Lindenstrauss projection must be a pure, replayable
    function of its version stamp and dimensions."""

    def test_replayable_across_independent_constructions(self) -> None:
        left = FrozenProjection(version="jl-gauss-v1", input_dim=768, output_dim=16)
        right = FrozenProjection(version="jl-gauss-v1", input_dim=768, output_dim=16)
        np.testing.assert_array_equal(left.matrix, right.matrix)

    def test_projects_single_vector_to_output_dim(self) -> None:
        projection = FrozenProjection(
            version="jl-gauss-v1", input_dim=768, output_dim=16
        )
        raw = np.ones(768, dtype=float)
        reduced = projection.project(raw)
        assert reduced.shape == (16,)

    def test_projects_batch_row_wise(self) -> None:
        projection = FrozenProjection(
            version="jl-gauss-v1", input_dim=768, output_dim=16
        )
        rng = np.random.default_rng(0)
        batch = rng.normal(size=(5, 768))
        reduced = projection.project(batch)
        assert reduced.shape == (5, 16)
        for index in range(5):
            np.testing.assert_allclose(reduced[index], projection.project(batch[index]))

    def test_version_bump_changes_the_projection(self) -> None:
        first = FrozenProjection(version="jl-gauss-v1", input_dim=768, output_dim=16)
        second = FrozenProjection(version="jl-gauss-v2", input_dim=768, output_dim=16)
        assert not np.allclose(first.matrix, second.matrix)

    def test_rejects_input_of_the_wrong_dimension(self) -> None:
        projection = FrozenProjection(
            version="jl-gauss-v1", input_dim=768, output_dim=16
        )
        with pytest.raises(ValueError):
            projection.project(np.ones(100, dtype=float))

    def test_approximately_preserves_relative_norms(self) -> None:
        # Johnson-Lindenstrauss: the reduced geometry should track the original
        # up to a modest distortion, so semantically close cards stay close.
        projection = FrozenProjection(
            version="jl-gauss-v1", input_dim=768, output_dim=256
        )
        rng = np.random.default_rng(1)
        a = rng.normal(size=768)
        b = rng.normal(size=768)
        original = float(np.linalg.norm(a - b))
        reduced = float(np.linalg.norm(projection.project(a) - projection.project(b)))
        assert reduced == pytest.approx(original, rel=0.25)

    def test_rejects_nonpositive_dimensions(self) -> None:
        with pytest.raises(ValueError):
            FrozenProjection(version="jl-gauss-v1", input_dim=0, output_dim=16)
        with pytest.raises(ValueError):
            FrozenProjection(version="jl-gauss-v1", input_dim=768, output_dim=0)


class TestByteIdenticalSeam:
    """The disabled prior must leave the model-config identity untouched."""

    def test_disabled_prior_is_absent_from_the_feature_config_payload(self) -> None:
        # ``model_config_hash`` dumps the feature config with ``exclude_none``;
        # a None embedding prior must not appear, so the hash is unchanged.
        config = FeatureConfig(behavior_keys=BEHAVIOR_KEYS)
        payload = config.model_dump(mode="json", exclude_none=True)
        assert "embedding_prior" not in payload

    def test_enabled_prior_enters_the_feature_config_payload(self) -> None:
        config = FeatureConfig(
            behavior_keys=BEHAVIOR_KEYS, embedding_prior=EmbeddingPriorConfig()
        )
        payload = config.model_dump(mode="json", exclude_none=True)
        assert "embedding_prior" in payload

    def test_exclude_none_drops_only_the_embedding_prior_key(self) -> None:
        # The disabled-prior hash equals the pre-embedding hash *because*
        # ``embedding_prior`` is the sole None-valued field ``exclude_none``
        # removes. A future nullable field would be silently pruned too and
        # would break the byte-identical control without touching this seam --
        # this pins the invariant so that regression fails loud here first.
        config = FeatureConfig(behavior_keys=BEHAVIOR_KEYS)
        full = config.model_dump(mode="json")
        pruned = config.model_dump(mode="json", exclude_none=True)
        assert set(full) - set(pruned) == {"embedding_prior"}

    def test_disabled_prior_leaves_the_design_width_unchanged(
        self, revisions: tuple[CardSnapshot, CardSnapshot]
    ) -> None:
        off = _space(None, revisions)
        assert off.embedding_effect_dim == 0
        expected = off.baseline_dim + off.card_effect_slice.stop
        assert off.outcome_dim == expected


class TestAugmentedFeatureSpace:
    def test_enabling_the_prior_adds_m_times_d_shared_columns(
        self, revisions: tuple[CardSnapshot, CardSnapshot]
    ) -> None:
        dimension = 8
        embeddings = {card.bank_card_id: np.zeros(dimension) for card in revisions}
        off = _space(None, revisions)
        on = _space(
            EmbeddingPriorConfig(dimension=dimension), revisions, embeddings=embeddings
        )
        d = off.card_context_dim
        assert on.embedding_effect_dim == dimension * d
        assert on.outcome_dim == off.outcome_dim + dimension * d

    def test_embedding_block_is_the_outer_product_of_context_and_phi(
        self,
        revisions: tuple[CardSnapshot, CardSnapshot],
        evolution_context,
    ) -> None:
        dimension = 4
        good, bad = revisions
        phi_good = np.array([0.5, -0.2, 0.1, 0.3])
        embeddings = {
            good.bank_card_id: phi_good,
            bad.bank_card_id: np.zeros(dimension),
        }
        on = _space(
            EmbeddingPriorConfig(dimension=dimension), revisions, embeddings=embeddings
        )
        row = on.design(good, evolution_context, True)
        context = on.context_features(evolution_context)
        d = on.card_context_dim
        block = row[
            on.baseline_dim + on.embedding_effect_slice.start : on.baseline_dim
            + on.embedding_effect_slice.stop
        ]
        expected = np.outer(context / np.sqrt(d), phi_good).ravel()
        np.testing.assert_allclose(block, expected)

    def test_control_row_zeros_the_embedding_block(
        self,
        revisions: tuple[CardSnapshot, CardSnapshot],
        evolution_context,
    ) -> None:
        dimension = 4
        good, bad = revisions
        embeddings = {
            good.bank_card_id: np.ones(dimension),
            bad.bank_card_id: np.ones(dimension),
        }
        on = _space(
            EmbeddingPriorConfig(dimension=dimension), revisions, embeddings=embeddings
        )
        row = on.design(good, evolution_context, False)
        block = row[
            on.baseline_dim + on.embedding_effect_slice.start : on.baseline_dim
            + on.embedding_effect_slice.stop
        ]
        np.testing.assert_array_equal(block, 0.0)

    def test_embedding_false_zeros_the_block_for_the_safety_head(
        self,
        revisions: tuple[CardSnapshot, CardSnapshot],
        evolution_context,
    ) -> None:
        dimension = 4
        good, bad = revisions
        embeddings = {
            good.bank_card_id: np.ones(dimension),
            bad.bank_card_id: np.ones(dimension),
        }
        on = _space(
            EmbeddingPriorConfig(dimension=dimension), revisions, embeddings=embeddings
        )
        treated_with = on.design(good, evolution_context, True, embedding=True)
        treated_without = on.design(good, evolution_context, True, embedding=False)
        block = slice(
            on.baseline_dim + on.embedding_effect_slice.start,
            on.baseline_dim + on.embedding_effect_slice.stop,
        )
        assert np.any(treated_with[block] != 0.0)
        np.testing.assert_array_equal(treated_without[block], 0.0)
        # Every non-embedding column stays byte-identical between the two heads.
        stripped_with = np.delete(treated_with, np.r_[block])
        stripped_without = np.delete(treated_without, np.r_[block])
        np.testing.assert_array_equal(stripped_with, stripped_without)

    def test_prior_variance_uses_the_reward_scale_on_the_embedding_block(
        self, revisions: tuple[CardSnapshot, CardSnapshot]
    ) -> None:
        dimension = 4
        embeddings = {card.bank_card_id: np.zeros(dimension) for card in revisions}
        on = _space(
            EmbeddingPriorConfig(dimension=dimension, reward_prior_sd=0.5),
            revisions,
            embeddings=embeddings,
        )
        variance = on.prior_variance(
            baseline_sd=0.75, shared_effect_sd=0.35, card_effect_sd=0.25
        )
        block = variance[
            on.baseline_dim + on.embedding_effect_slice.start : on.baseline_dim
            + on.embedding_effect_slice.stop
        ]
        np.testing.assert_allclose(block, 0.5**2)

    def test_missing_embedding_is_rejected_when_the_prior_is_enabled(
        self, revisions: tuple[CardSnapshot, CardSnapshot]
    ) -> None:
        with pytest.raises(ValueError):
            _space(EmbeddingPriorConfig(dimension=4), revisions, embeddings={})

    def test_embedding_of_the_wrong_length_is_rejected(
        self, revisions: tuple[CardSnapshot, CardSnapshot]
    ) -> None:
        embeddings = {card.bank_card_id: np.zeros(3) for card in revisions}
        with pytest.raises(ValueError):
            _space(EmbeddingPriorConfig(dimension=4), revisions, embeddings=embeddings)

    def test_non_finite_embedding_is_rejected_when_the_prior_is_enabled(
        self, revisions: tuple[CardSnapshot, CardSnapshot]
    ) -> None:
        # A truncated or failed embedder can return NaN/Inf; it would poison the
        # reward design and yield a NaN posterior. Fail closed like the sibling
        # shape/missing guards.
        good, bad = revisions
        embeddings = {
            good.bank_card_id: np.array([np.nan, 1.0, 1.0, 1.0]),
            bad.bank_card_id: np.zeros(4),
        }
        with pytest.raises(ValueError):
            _space(EmbeddingPriorConfig(dimension=4), revisions, embeddings=embeddings)

    def test_absorbed_alias_never_overrides_the_survivor_embedding(
        self, evolution_context
    ) -> None:
        # A lineage's survivor and an absorbed alias can both carry a vector; the
        # survivor's own must win regardless of mapping order, or the resolved
        # feature would be non-deterministic.
        survivor = CardSnapshot.from_card(
            Card(
                id="survivor",
                task_key="task",
                description="same treatment",
                absorbed_ids=("old",),
            )
        )
        dimension = 4
        survivor_phi = np.arange(1.0, 1.0 + dimension)
        alias_phi = -survivor_phi
        for provided in (
            {"survivor": survivor_phi, "old": alias_phi},
            {"old": alias_phi, "survivor": survivor_phi},
        ):
            space = _space(
                EmbeddingPriorConfig(dimension=dimension),
                (survivor,),
                embeddings=provided,
            )
            start = space.baseline_dim + space.embedding_effect_slice.start
            stop = space.baseline_dim + space.embedding_effect_slice.stop
            block = space.design(survivor, evolution_context, True)[start:stop]
            context = space.context_features(evolution_context)
            expected = np.outer(
                context / np.sqrt(space.card_context_dim), survivor_phi
            ).ravel()
            np.testing.assert_allclose(block, expected)


def _reward_observations(
    context,
    card: CardSnapshot,
    *,
    effect: float,
    start_ordinal: int,
    per_arm: int = 40,
    seed: int = 3,
) -> tuple[CausalObservation, ...]:
    rng = np.random.default_rng(seed)
    rows: list[CausalObservation] = []
    ordinal = start_ordinal
    for treatment in (False, True):
        for _ in range(per_arm):
            value = float(rng.normal(effect if treatment else 0.0, 0.05))
            rows.append(
                CausalObservation(
                    decision_id=f"decision-{ordinal}",
                    event_ordinal=ordinal,
                    card=card,
                    context=context,
                    treatment=treatment,
                    card_used=treatment,
                    offer_propensity=0.5,
                    proposal_propensity=0.5,
                    joint_action_propensity=0.25,
                    status="outcome",
                    measurement=OutcomeMeasurement(value=value, se=None, kind="scalar"),
                    reward_q_hat_control=0.0,
                    reward_q_hat_treated=0.0,
                    risk_q_hat_control=0.05,
                    risk_q_hat_treated=0.05,
                )
            )
            ordinal += 1
    return tuple(rows)


class TestColdStartAndSafetyInvariance:
    """The embedding prior must lift an unobserved card toward its semantic
    neighbours while leaving the safety head byte-identical."""

    def _cards(self) -> tuple[CardSnapshot, CardSnapshot, CardSnapshot]:
        return (
            CardSnapshot.from_card(
                Card(id="helpful", task_key="task", description="use the strong lever")
            ),
            CardSnapshot.from_card(
                Card(id="weak", task_key="task", description="use the flat lever")
            ),
            CardSnapshot.from_card(
                Card(id="cold", task_key="task", description="use the untried lever")
            ),
        )

    def _fit(self, evolution_context, *, embedding_prior, embeddings):
        helpful, weak, cold = self._cards()
        observations = (
            *_reward_observations(
                evolution_context, helpful, effect=0.25, start_ordinal=0, seed=1
            ),
            *_reward_observations(
                evolution_context, weak, effect=-0.05, start_ordinal=200, seed=2
            ),
        )
        model = HierarchicalTerminalUtilityPosterior(
            feature_map=HierarchicalFeatureMap(
                config=FeatureConfig(
                    behavior_keys=BEHAVIOR_KEYS, embedding_prior=embedding_prior
                )
            ),
            config=TerminalUtilityPosteriorConfig(),
        )
        fitted = model.fit(
            observations,
            (helpful, weak, cold),
            card_embeddings=embeddings,
        )
        return fitted, (helpful, weak, cold)

    def test_embedding_prior_lifts_the_unobserved_card_toward_its_neighbour(
        self, evolution_context
    ) -> None:
        dimension = 6
        rng = np.random.default_rng(11)
        # Cold card shares the helpful card's semantic direction; the weak card
        # is orthogonal.
        phi_helpful = rng.normal(size=dimension)
        phi_cold = phi_helpful + 0.05 * rng.normal(size=dimension)
        phi_weak = rng.normal(size=dimension)

        fitted_off, (_, _, cold_off) = self._fit(
            evolution_context, embedding_prior=None, embeddings=None
        )
        prediction_rng = np.random.default_rng(99)
        effect_off = fitted_off.prediction(
            cold_off,
            evolution_context,
            prediction_rng,
            samples=2048,
            max_treated_invalid_probability=0.25,
            max_incremental_invalid_probability=0.1,
            safety_alpha=0.1,
        ).usable_effect_mean

        helpful, weak, cold = self._cards()
        embeddings = {
            helpful.bank_card_id: phi_helpful,
            weak.bank_card_id: phi_weak,
            cold.bank_card_id: phi_cold,
        }
        fitted_on, (_, _, cold_on) = self._fit(
            evolution_context,
            embedding_prior=EmbeddingPriorConfig(dimension=dimension),
            embeddings=embeddings,
        )
        prediction_rng = np.random.default_rng(99)
        effect_on = fitted_on.prediction(
            cold_on,
            evolution_context,
            prediction_rng,
            samples=2048,
            max_treated_invalid_probability=0.25,
            max_incremental_invalid_probability=0.1,
            safety_alpha=0.1,
        ).usable_effect_mean

        # With no observations of the cold card, the zero-mean prior leaves it at
        # the shared mean; the embedding prior borrows the helpful neighbour's
        # positive effect.
        assert effect_on > effect_off

    def test_embedding_prior_pulls_a_cold_card_toward_a_negative_neighbour(
        self, evolution_context
    ) -> None:
        # Mirror of the lift test: when the cold card's direction matches the
        # *weak* neighbour (below the shared mean), the prior must pull the
        # prediction down, not up. This pins that the borrowed signal is
        # directional, not a blanket upward nudge.
        dimension = 6
        rng = np.random.default_rng(23)
        phi_helpful = rng.normal(size=dimension)
        phi_weak = rng.normal(size=dimension)
        phi_cold = phi_weak + 0.05 * rng.normal(size=dimension)

        fitted_off, (_, _, cold_off) = self._fit(
            evolution_context, embedding_prior=None, embeddings=None
        )
        prediction_rng = np.random.default_rng(99)
        effect_off = fitted_off.prediction(
            cold_off,
            evolution_context,
            prediction_rng,
            samples=2048,
            max_treated_invalid_probability=0.25,
            max_incremental_invalid_probability=0.1,
            safety_alpha=0.1,
        ).usable_effect_mean

        helpful, weak, cold = self._cards()
        embeddings = {
            helpful.bank_card_id: phi_helpful,
            weak.bank_card_id: phi_weak,
            cold.bank_card_id: phi_cold,
        }
        fitted_on, (_, _, cold_on) = self._fit(
            evolution_context,
            embedding_prior=EmbeddingPriorConfig(dimension=dimension),
            embeddings=embeddings,
        )
        prediction_rng = np.random.default_rng(99)
        effect_on = fitted_on.prediction(
            cold_on,
            evolution_context,
            prediction_rng,
            samples=2048,
            max_treated_invalid_probability=0.25,
            max_incremental_invalid_probability=0.1,
            safety_alpha=0.1,
        ).usable_effect_mean

        assert effect_on < effect_off

    def test_safety_linear_predictor_is_byte_identical_across_the_seam(
        self, evolution_context
    ) -> None:
        dimension = 6
        rng = np.random.default_rng(7)
        fitted_off, cards_off = self._fit(
            evolution_context, embedding_prior=None, embeddings=None
        )
        helpful, weak, cold = self._cards()
        embeddings = {
            card.bank_card_id: rng.normal(size=dimension)
            for card in (helpful, weak, cold)
        }
        fitted_on, cards_on = self._fit(
            evolution_context,
            embedding_prior=EmbeddingPriorConfig(dimension=dimension),
            embeddings=embeddings,
        )
        for card_off, card_on in zip(cards_off, cards_on, strict=True):
            for treatment in (False, True):
                design_off = fitted_off.space.design(
                    card_off, evolution_context, treatment, embedding=False
                )
                design_on = fitted_on.space.design(
                    card_on, evolution_context, treatment, embedding=False
                )
                predictor_off = float(design_off @ fitted_off.safety.mean)
                predictor_on = float(design_on @ fitted_on.safety.mean)
                assert predictor_on == pytest.approx(predictor_off, abs=1e-9)

    def test_fit_needs_embeddings_for_every_observed_lineage_not_just_candidates(
        self, evolution_context
    ) -> None:
        # An observed card that is no longer a current candidate still contributes
        # rows carrying the shared embedding columns, so ``fit`` needs its
        # projection. This is why the provider reduces observations + lineage +
        # candidates, not candidates alone.
        helpful, weak, cold = self._cards()
        observations = (
            *_reward_observations(
                evolution_context, helpful, effect=0.25, start_ordinal=0, seed=1
            ),
            *_reward_observations(
                evolution_context, weak, effect=-0.05, start_ordinal=200, seed=2
            ),
        )
        model = HierarchicalTerminalUtilityPosterior(
            feature_map=HierarchicalFeatureMap(
                config=FeatureConfig(
                    behavior_keys=BEHAVIOR_KEYS,
                    embedding_prior=EmbeddingPriorConfig(dimension=4),
                )
            ),
            config=TerminalUtilityPosteriorConfig(),
        )
        rng = np.random.default_rng(5)
        full = {card.bank_card_id: rng.normal(size=4) for card in (helpful, weak, cold)}
        # cold is the only candidate; helpful/weak survive only as observations.
        model.fit(observations, (cold,), card_embeddings=full)
        missing_observed = {cold.bank_card_id: full[cold.bank_card_id]}
        with pytest.raises(ValueError):
            model.fit(observations, (cold,), card_embeddings=missing_observed)


class _CountingEmbedder(CardEmbedder):
    """Deterministic fake embedder recording every batch it is asked to embed."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self.batches: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts) -> np.ndarray:
        texts = list(texts)
        self.batches.append(tuple(texts))
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rows.append(np.random.default_rng(seed).normal(size=self._dimension))
        return np.asarray(rows, dtype=float)


class TestCardEmbeddingReducer:
    """Reduce candidate card payloads to frozen projected features, cached by text."""

    def _cards(self) -> tuple[CardSnapshot, CardSnapshot]:
        return (
            CardSnapshot.from_card(
                Card(id="a", task_key="task", description="strong lever")
            ),
            CardSnapshot.from_card(
                Card(id="b", task_key="task", description="flat lever")
            ),
        )

    def _reducer(
        self, embedder: CardEmbedder, *, input_dim: int, output_dim: int
    ) -> tuple[CardEmbeddingReducer, FrozenProjection]:
        projection = FrozenProjection(
            version="jl-gauss-v1", input_dim=input_dim, output_dim=output_dim
        )
        return CardEmbeddingReducer(
            embedder=embedder, projection=projection
        ), projection

    def test_reduce_keys_by_bank_card_id_and_matches_projected_embedding(self) -> None:
        input_dim, output_dim = 32, 8
        embedder = _CountingEmbedder(input_dim)
        reducer, projection = self._reducer(
            embedder, input_dim=input_dim, output_dim=output_dim
        )
        cards = self._cards()

        reduced = reducer.reduce(cards)

        assert set(reduced) == {card.bank_card_id for card in cards}
        for card in cards:
            raw = embedder.embed([card.payload])
            unit = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
            expected = projection.project(unit)[0]
            np.testing.assert_allclose(reduced[card.bank_card_id], expected)
            assert reduced[card.bank_card_id].shape == (output_dim,)

    def test_reduce_caches_by_text_and_dedupes_within_a_batch(self) -> None:
        embedder = _CountingEmbedder(16)
        reducer, _ = self._reducer(embedder, input_dim=16, output_dim=4)
        a, b = self._cards()
        a_twin = CardSnapshot.from_card(
            Card(id="a_twin", task_key="task", description="strong lever")
        )
        # Same embedded text, distinct bank ids: one embed, one shared vector.
        assert a.payload == a_twin.payload
        assert a.bank_card_id != a_twin.bank_card_id

        first = reducer.reduce((a, b, a_twin))
        assert len(embedder.batches) == 1
        assert len(embedder.batches[0]) == 2
        np.testing.assert_array_equal(first[a.bank_card_id], first[a_twin.bank_card_id])

        second = reducer.reduce((a, b))
        assert len(embedder.batches) == 1  # fully cached, no new embed call
        np.testing.assert_array_equal(second[a.bank_card_id], first[a.bank_card_id])

    def test_reduce_of_an_empty_candidate_set_is_empty_and_embeds_nothing(self) -> None:
        embedder = _CountingEmbedder(8)
        reducer, _ = self._reducer(embedder, input_dim=8, output_dim=4)

        assert reducer.reduce(()) == {}
        assert embedder.batches == []

    def test_reduce_is_invariant_to_the_raw_embedding_norm(self) -> None:
        # The induced effect-prior width scales with ||phi||^2, so rescaling the
        # embedder's raw output must not move the reduced features: the prior
        # tracks semantic direction, never the embedder's arbitrary magnitude.
        input_dim, output_dim = 16, 4
        base = _CountingEmbedder(input_dim)

        class _Scaled(CardEmbedder):
            @property
            def dimension(self) -> int:
                return input_dim

            def embed(self, texts) -> np.ndarray:
                return 37.0 * base.embed(texts)

        reducer_a, _ = self._reducer(base, input_dim=input_dim, output_dim=output_dim)
        reducer_b, _ = self._reducer(
            _Scaled(), input_dim=input_dim, output_dim=output_dim
        )
        cards = self._cards()

        reduced_a = reducer_a.reduce(cards)
        reduced_b = reducer_b.reduce(cards)
        for card in cards:
            np.testing.assert_allclose(
                reduced_a[card.bank_card_id], reduced_b[card.bank_card_id]
            )
