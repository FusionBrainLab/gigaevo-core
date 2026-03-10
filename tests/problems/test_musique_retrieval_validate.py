from problems.chains.musique_retrieval.utils.failure_artifact import (
    build_failed_examples_artifact,
)


def test_failure_artifact_includes_summary_and_examples():
    dataset = [
        {"task_id": "t1", "question": "Who is alpha?"},
        {"task_id": "t2", "question": "Who is beta?"},
        {"task_id": "t3", "question": "Who is gamma?"},
    ]
    targets = [["alpha"], ["beta"], ["gamma"]]
    predictions = ["alpha", None, "delta"]

    artifact = build_failed_examples_artifact(
        dataset,
        targets,
        predictions,
        fitness=1 / 3,
        extraction_failures=1 / 3,
        max_examples=10,
    )

    assert "Overall score (fitness / EM): 0.3333" in artifact
    assert "Extraction failure rate: 0.3333" in artifact
    assert "Failed samples: 2" in artifact
    assert "task_id=t2 | failure=extraction_failure" in artifact
    assert "task_id=t3 | failure=mismatch" in artifact


def test_failure_artifact_handles_no_failures():
    dataset = [{"task_id": "t1", "question": "Who is alpha?"}]
    targets = [["alpha"]]
    predictions = ["Alpha"]

    artifact = build_failed_examples_artifact(
        dataset,
        targets,
        predictions,
        fitness=1.0,
        extraction_failures=0.0,
        max_examples=10,
    )

    assert "Failed samples: 0" in artifact
    assert "No failed samples." in artifact
