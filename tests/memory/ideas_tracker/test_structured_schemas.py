from pydantic import ValidationError
import pytest

from gigaevo.memory.ideas_tracker.schemas import (
    ClassifyExtResponse,
    ClusterPartitionResponse,
    PresentIdeaRef,
    RepresentativeChoiceResponse,
    SynthesisedDescription,
    UpdatedIdeaRef,
)


class TestIdeaIdNormalization:
    def test_present_ref_strips_brackets(self):
        ref = PresentIdeaRef(idea_id="[a1b2c3]", sequence=2)
        assert ref.idea_id == "a1b2c3"

    def test_present_ref_strips_whitespace_and_brackets(self):
        ref = PresentIdeaRef(idea_id="  [ a1b2c3 ]  ", sequence=1)
        assert ref.idea_id == "a1b2c3"

    def test_updated_ref_strips_brackets(self):
        ref = UpdatedIdeaRef(idea_id="[xyz]", sequence=3, text="richer text")
        assert ref.idea_id == "xyz"

    def test_plain_id_unchanged(self):
        ref = PresentIdeaRef(idea_id="a1b2c3", sequence=1)
        assert ref.idea_id == "a1b2c3"


class TestExtraKeysForbidden:
    def test_classify_response_rejects_extra_key(self):
        with pytest.raises(ValidationError):
            ClassifyExtResponse(
                new_ideas=[], present_ideas=[], updated_ideas=[], bogus=1
            )

    def test_partition_rejects_extra_key(self):
        with pytest.raises(ValidationError):
            ClusterPartitionResponse(included=[1], rejected=[], split_suggestion=[[1]])


class TestRoundtrip:
    def test_classify_response_json_roundtrip(self):
        original = ClassifyExtResponse(
            new_ideas=["use Sobol init"],
            present_ideas=[PresentIdeaRef(idea_id="abc123", sequence=2)],
            updated_ideas=[
                UpdatedIdeaRef(idea_id="def456", sequence=3, text="maxiter 5000→20000")
            ],
        )
        parsed = ClassifyExtResponse.model_validate_json(original.model_dump_json())
        assert parsed == original

    def test_partition_roundtrip(self):
        original = ClusterPartitionResponse(included=[1, 2], rejected=[3])
        parsed = ClusterPartitionResponse.model_validate_json(
            original.model_dump_json()
        )
        assert parsed == original

    def test_representative_roundtrip(self):
        parsed = RepresentativeChoiceResponse.model_validate_json(
            '{"representative_index": 4}'
        )
        assert parsed.representative_index == 4

    def test_description_roundtrip(self):
        parsed = SynthesisedDescription.model_validate_json(
            '{"description": "Increases SLSQP maxiter."}'
        )
        assert parsed.description == "Increases SLSQP maxiter."


class TestRequiredFields:
    def test_classify_response_requires_all_lists(self):
        with pytest.raises(ValidationError):
            ClassifyExtResponse.model_validate_json('{"new_ideas": []}')

    def test_updated_ref_requires_text(self):
        with pytest.raises(ValidationError):
            UpdatedIdeaRef(idea_id="abc", sequence=1)
