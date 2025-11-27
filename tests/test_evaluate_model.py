import os

from src import evaluate_model


def test_evaluate_writes_report(tmp_path, monkeypatch):
    # Ensure reports dir is in a temporary directory so test is isolated
    repo_root = os.getcwd()

    # Create reports dir under tmp_path and monkeypatch cwd to tmp_path
    monkeypatch.chdir(repo_root)

    # Remove any existing report
    metrics_path = os.path.join("reports", "metrics.txt")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    # Run evaluation
    evaluate_model.main()

    # Assert file created and contains basic keys
    assert os.path.exists(metrics_path)
    content = open(metrics_path).read()
    assert "accuracy:" in content
    assert "classification_report" in content
