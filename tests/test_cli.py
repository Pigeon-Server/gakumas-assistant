import pytest

from src.cli.main import CLIError, _normalize_global_option_order, _parse_config_value, build_parser


def test_cli_parser_supports_task_run_from():
    parser = build_parser()

    args = parser.parse_args(["tasks", "run", "--from", "dispatch_work"])

    assert args.command == "tasks"
    assert args.tasks_command == "run"
    assert args.from_task is True
    assert args.task_id == "dispatch_work"


def test_parse_config_bool_value():
    assert _parse_config_value("true", bool) is True
    assert _parse_config_value("否", bool) is False


def test_parse_config_list_requires_json():
    assert _parse_config_value('["a", "b"]', list) == ["a", "b"]
    with pytest.raises(CLIError):
        _parse_config_value("a,b", list)


def test_normalize_global_option_order():
    assert _normalize_global_option_order(["status", "--quiet"]) == ["--quiet", "status"]
    assert _normalize_global_option_order(["tasks", "list", "--log-level", "DEBUG"]) == [
        "--log-level",
        "DEBUG",
        "tasks",
        "list",
    ]
