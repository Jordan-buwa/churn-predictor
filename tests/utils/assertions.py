# tests/utils/assertions.py
from fastapi import HTTPException
from pathlib import Path
from datetime import datetime


# Add to tests/utils/assertions.py
from pathlib import Path


def assert_path_exists(path):
    """
    Assert that a file or directory exists at the given path.
    """
    p = Path(path)
    assert p.exists(), f"Path does not exist: {path}"


def assert_status(response_json, expected_statuses, message=None):
    """
    Assert that response_json["status"] matches one of the expected statuses.

    Args:
        response_json (dict): The API response.
        expected_statuses (list[str]): Accepted values for "status".
        message (str, optional): Additional message on failure.
    """
    actual = response_json.get("status")
    assert actual in expected_statuses, (
        message or f"Expected one of {expected_statuses}, got '{actual}'"
    )


def assert_job_entry(job, expected_fields=None):
    """
    Assert that a job dict has the expected structure and fields.
    """
    expected_fields = expected_fields or [
        "job_id", "status", "model_type", "script_path",
        "started_at", "completed_at", "model_path", "error", "logs"
    ]
    for field in expected_fields:
        assert field in job, f"Missing field in job entry: {field}"


def assert_isoformat(date_string):
    """
    Assert that a string is a valid ISO 8601 datetime.
    """
    try:
        datetime.fromisoformat(date_string)
    except ValueError:
        raise AssertionError(
            f"String is not a valid ISO format: {date_string}")


def assert_file_exists_with_extensions(file_path, extensions):
    """
    Assert that a file exists and has one of the specified extensions.

    Args:
        file_path (str or Path): Path to the file.
        extensions (list[str]): List of allowed extensions, e.g., ['.joblib', '.pkl'].

    Raises:
        AssertionError: If the file does not exist or has an invalid extension.
    """
    path = Path(file_path)
    assert path.exists(), f"File does not exist: {file_path}"
    assert path.suffix in extensions, f"File {file_path} does not have one of the expected extensions: {extensions}"


# Add to tests/utils/assertions.py


def assert_http_exception(exc: HTTPException, expected_status: int = None, expected_detail: str = None):
    """
    Assert that an HTTPException has the expected status code and detail.
    """
    if expected_status is not None:
        assert exc.status_code == expected_status, f"Expected status {expected_status}, got {exc.status_code}"
    if expected_detail is not None:
        assert expected_detail.lower() in str(exc.detail).lower(
        ), f"Expected detail to contain '{expected_detail}', got '{exc.detail}'"
