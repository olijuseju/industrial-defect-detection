import json
import logging
import sys

import pytest

from defect_detection.logging_config import (
    JsonFormatter,
    _ProjectHandler,
    configure_logging,
    request_id_var,
)


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers, root.level = handlers, level


def make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord("svc", logging.INFO, __file__, 1, "Hello %s", ("world",), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_puts_extra_fields_at_the_top_level():
    entry = json.loads(JsonFormatter().format(make_record(defective=True, detections=2)))

    assert entry["message"] == "Hello world"
    assert entry["level"] == "INFO" and entry["logger"] == "svc"
    assert entry["defective"] is True and entry["detections"] == 2
    assert entry["timestamp"].endswith("+00:00")


def test_json_formatter_omits_empty_and_cosmetic_fields_and_survives_odd_values(tmp_path):
    record = make_record(request_id=None, path=tmp_path, color_message="\x1b[36mHello")
    entry = json.loads(JsonFormatter().format(record))

    assert "request_id" not in entry and "color_message" not in entry
    assert entry["path"] == str(tmp_path)


def test_json_formatter_includes_the_traceback():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord("svc", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())

    entry = json.loads(JsonFormatter().format(record))

    assert "RuntimeError: boom" in entry["exception"]


def test_configure_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    configure_logging()
    configure_logging()

    ours = [h for h in logging.getLogger().handlers if isinstance(h, _ProjectHandler)]
    assert len(ours) == 1


def test_environment_overrides_format_and_level(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "warning")

    configure_logging()

    root = logging.getLogger()
    ours = [h for h in root.handlers if isinstance(h, _ProjectHandler)]
    assert isinstance(ours[0].formatter, JsonFormatter)
    assert root.level == logging.WARNING


def test_unknown_format_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOG_FORMAT", "xml")

    with pytest.raises(ValueError, match="LOG_FORMAT"):
        configure_logging()


def test_records_carry_the_current_request_id(caplog: pytest.LogCaptureFixture):
    configure_logging()
    token = request_id_var.set("abc123")
    try:
        with caplog.at_level(logging.INFO):
            logging.getLogger("anything").info("inside a request")
    finally:
        request_id_var.reset(token)

    assert caplog.records[-1].request_id == "abc123"
