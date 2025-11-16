"""Tests for logging configuration helpers in core.acm_main."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from core import acm_main


def _make_args(**overrides):
    defaults = dict(
        equip="FD_FAN",
        artifact_root="artifacts",
        config=None,
        train_csv=None,
        score_csv=None,
        log_level=None,
        log_format=None,
        log_file=None,
        log_module_level=[],
        disable_sql_logging=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_configure_logging_cli_overrides():
    logging_cfg = {
        "level": "INFO",
        "format": "text",
        "module_levels": {"core.acm_main": "WARNING"},
    }
    args = _make_args(
        log_level="DEBUG",
        log_format="json",
        log_file="artifacts/acm.log",
        log_module_level=["scripts.sql_batch_runner=ERROR"],
        disable_sql_logging=True,
    )

    with mock.patch.object(acm_main.Console, "set_level") as mock_set_level, \
         mock.patch.object(acm_main.Console, "set_format") as mock_set_format, \
         mock.patch.object(acm_main.Console, "set_output") as mock_set_output, \
         mock.patch.object(acm_main.Console, "clear_module_levels") as mock_clear_levels, \
         mock.patch.object(acm_main.Console, "set_module_level") as mock_set_module:

        settings = acm_main._configure_logging(logging_cfg, args)

    mock_set_level.assert_called_with("DEBUG")
    mock_set_format.assert_called_with("json")
    mock_set_output.assert_called()
    mock_clear_levels.assert_called_once()
    set_calls = [call.args for call in mock_set_module.call_args_list]
    assert ("core.acm_main", "WARNING") in set_calls
    assert ("scripts.sql_batch_runner", "ERROR") in set_calls
    assert settings["enable_sql_logging"] is False


def test_configure_logging_config_defaults():
    logging_cfg = {
        "level": "WARNING",
        "format": "json",
        "file": "artifacts/logs/acm.log",
        "module_levels": ["core.acm_main=ERROR"],
        "enable_sql_sink": True,
    }
    args = _make_args()

    with mock.patch.object(acm_main.Console, "set_level") as mock_set_level, \
         mock.patch.object(acm_main.Console, "set_format") as mock_set_format, \
         mock.patch.object(acm_main.Console, "set_output") as mock_set_output, \
         mock.patch.object(acm_main.Console, "clear_module_levels") as mock_clear_levels, \
         mock.patch.object(acm_main.Console, "set_module_level") as mock_set_module:

        settings = acm_main._configure_logging(logging_cfg, args)

    mock_set_level.assert_called_with("WARNING")
    mock_set_format.assert_called_with("json")
    mock_set_output.assert_called_once()
    mock_clear_levels.assert_called_once()
    mock_set_module.assert_called_with("core.acm_main", "ERROR")
    assert settings["enable_sql_logging"] is True
