"""Offline tests for sequential ERA5 range orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
SCRIPT = SCRIPTS / "download_era5_range.py"


@pytest.fixture(scope="module")
def range_downloader():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("era5_range_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_complete_day(module, output_root: Path, requested_date: str) -> Path:
    directory = module.dataset_directory(output_root, requested_date)
    directory.mkdir(parents=True, exist_ok=True)
    for name in module.REQUIRED_MANIFEST_FILES[:2]:
        (directory / name).write_bytes(name.encode("utf-8"))
    (directory / "request.json").write_text("{}\n", encoding="utf-8")
    (directory / "metadata.json").write_text("{}\n", encoding="utf-8")
    lines = [
        f"{module.sha256_file(directory / name)}  {name}" for name in module.REQUIRED_MANIFEST_FILES
    ]
    (directory / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def test_inclusive_and_one_day_date_ranges(range_downloader) -> None:
    assert range_downloader.date_range("2024-01-01", "2024-01-03") == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert range_downloader.date_range("2024-01-02", "2024-01-02") == ["2024-01-02"]


@pytest.mark.parametrize("start, end", [("2024-02-30", "2024-03-01"), ("2024-01-01", "2024-02-30")])
def test_invalid_range_dates_fail(range_downloader, start: str, end: str) -> None:
    with pytest.raises(ValueError, match="Invalid --date"):
        range_downloader.date_range(start, end)


def test_reverse_range_fails(range_downloader) -> None:
    with pytest.raises(ValueError, match="must not be later"):
        range_downloader.date_range("2024-01-03", "2024-01-01")


def test_hourly_times_are_stable_and_complete(range_downloader) -> None:
    assert range_downloader.HOURLY_TIMES == tuple(f"{hour:02d}:00" for hour in range(24))


def test_dry_run_creates_no_directory_or_cds_import(tmp_path, range_downloader, monkeypatch, capsys) -> None:
    output_root = tmp_path / "raw"
    monkeypatch.delitem(sys.modules, "cdsapi", raising=False)
    assert range_downloader.main(
        ["--start-date", "2024-01-01", "--end-date", "2024-01-03", "--output-root", str(output_root), "--dry-run"]
    ) == 0
    assert not output_root.exists()
    assert "cdsapi" not in sys.modules
    assert "Hourly timestamps" in capsys.readouterr().out


def test_missing_day_is_planned_for_download(tmp_path, range_downloader) -> None:
    plan = range_downloader.plan_range(
        start_date="2024-01-01", end_date="2024-01-01", output_root=tmp_path, overwrite=False
    )
    assert plan[0]["status"] == "planned"
    assert plan[0]["action"] == "download"


def test_complete_verified_day_is_safely_skipped(tmp_path, range_downloader) -> None:
    _make_complete_day(range_downloader, tmp_path, "2024-01-01")
    plan = range_downloader.plan_range(
        start_date="2024-01-01", end_date="2024-01-01", output_root=tmp_path, overwrite=False
    )
    assert plan[0]["status"] == "skipped"
    assert plan[0]["action"] == "skip-existing"
    assert plan[0]["checksum_status"] == "passed"


def test_missing_required_file_is_not_complete(tmp_path, range_downloader) -> None:
    directory = range_downloader.dataset_directory(tmp_path, "2024-01-01")
    directory.mkdir()
    result = range_downloader.check_complete_day(directory)
    assert not result.complete
    assert result.checksum_status == "missing"


def test_invalid_checksum_and_malformed_manifest_are_rejected(tmp_path, range_downloader) -> None:
    directory = _make_complete_day(range_downloader, tmp_path, "2024-01-01")
    manifest = directory / "SHA256SUMS.txt"
    manifest.write_text("0" * 64 + "  request.json\n", encoding="utf-8")
    assert not range_downloader.check_complete_day(directory).complete
    manifest.write_text("not a manifest\n", encoding="utf-8")
    result = range_downloader.check_complete_day(directory)
    assert not result.complete
    assert "Malformed" in str(result.error)


def test_checksum_path_traversal_is_rejected(tmp_path, range_downloader) -> None:
    directory = _make_complete_day(range_downloader, tmp_path, "2024-01-01")
    (directory / "SHA256SUMS.txt").write_text("0" * 64 + "  ../outside.txt\n", encoding="utf-8")
    result = range_downloader.check_complete_day(directory)
    assert not result.complete
    assert "Unsafe SHA256SUMS path" in str(result.error)


def test_resume_skips_complete_days_and_passes_all_24_times(tmp_path, range_downloader) -> None:
    _make_complete_day(range_downloader, tmp_path, "2024-01-01")
    calls: list[dict[str, object]] = []

    def fake_daily(**kwargs):
        calls.append(kwargs)
        return _make_complete_day(range_downloader, kwargs["output_root"], kwargs["requested_date"])

    manifest = range_downloader.run_range(
        start_date="2024-01-01",
        end_date="2024-01-03",
        output_root=tmp_path,
        overwrite=False,
        keep_archive=False,
        continue_on_error=False,
        daily_downloader=fake_daily,
    )
    assert [call["requested_date"] for call in calls] == ["2024-01-02", "2024-01-03"]
    assert all(call["requested_times"] == range_downloader.HOURLY_TIMES for call in calls)
    assert manifest["summary"] == {
        **manifest["summary"],
        "requested_days": 3,
        "downloaded_days": 2,
        "verified_days": 3,
        "skipped_days": 1,
        "failed_days": 0,
        "completed": True,
    }


def test_default_stops_after_first_failed_day(tmp_path, range_downloader) -> None:
    calls: list[str] = []

    def failing_daily(**kwargs):
        calls.append(kwargs["requested_date"])
        raise RuntimeError("synthetic failure")

    manifest = range_downloader.run_range(
        start_date="2024-01-01",
        end_date="2024-01-03",
        output_root=tmp_path,
        overwrite=False,
        keep_archive=False,
        continue_on_error=False,
        daily_downloader=failing_daily,
    )
    assert calls == ["2024-01-01"]
    assert manifest["days"][0]["status"] == "failed"
    assert manifest["days"][1]["status"] == "planned"
    assert manifest["summary"]["completed"] is False


def test_existing_incomplete_day_fails_without_calling_daily_downloader(tmp_path, range_downloader) -> None:
    directory = range_downloader.dataset_directory(tmp_path, "2024-01-01")
    directory.mkdir()
    calls: list[dict[str, object]] = []

    def fake_daily(**kwargs):
        calls.append(kwargs)
        raise AssertionError("incomplete day must not be downloaded without --overwrite")

    manifest = range_downloader.run_range(
        start_date="2024-01-01",
        end_date="2024-01-01",
        output_root=tmp_path,
        overwrite=False,
        keep_archive=False,
        continue_on_error=False,
        daily_downloader=fake_daily,
    )
    assert calls == []
    assert manifest["days"][0]["status"] == "failed"
    assert "incomplete or invalid" in manifest["days"][0]["error"]


def test_continue_on_error_processes_later_days_and_returns_failed_manifest(tmp_path, range_downloader) -> None:
    calls: list[str] = []

    def sometimes_failing_daily(**kwargs):
        requested_date = kwargs["requested_date"]
        calls.append(requested_date)
        if requested_date == "2024-01-01":
            raise RuntimeError("synthetic failure")
        return _make_complete_day(range_downloader, kwargs["output_root"], requested_date)

    manifest = range_downloader.run_range(
        start_date="2024-01-01",
        end_date="2024-01-02",
        output_root=tmp_path,
        overwrite=False,
        keep_archive=False,
        continue_on_error=True,
        daily_downloader=sometimes_failing_daily,
    )
    assert calls == ["2024-01-01", "2024-01-02"]
    assert [entry["status"] for entry in manifest["days"]] == ["failed", "verified"]
    assert manifest["summary"]["failed_days"] == 1
    assert manifest["summary"]["completed"] is False


def test_main_returns_nonzero_when_run_manifest_has_failed_days(tmp_path, range_downloader, monkeypatch) -> None:
    monkeypatch.setattr(
        range_downloader,
        "run_range",
        lambda **_: {"summary": {"downloaded_days": 0, "verified_days": 0, "skipped_days": 0, "failed_days": 1, "completed": False}},
    )
    assert range_downloader.main(
        ["--start-date", "2024-01-01", "--end-date", "2024-01-01", "--output-root", str(tmp_path)]
    ) == 2


def test_overwrite_and_keep_archive_are_forwarded(tmp_path, range_downloader) -> None:
    _make_complete_day(range_downloader, tmp_path, "2024-01-01")
    calls: list[dict[str, object]] = []

    def fake_daily(**kwargs):
        calls.append(kwargs)
        return _make_complete_day(range_downloader, kwargs["output_root"], kwargs["requested_date"])

    manifest = range_downloader.run_range(
        start_date="2024-01-01",
        end_date="2024-01-01",
        output_root=tmp_path,
        overwrite=True,
        keep_archive=True,
        continue_on_error=False,
        daily_downloader=fake_daily,
    )
    assert calls[0]["overwrite"] is True
    assert calls[0]["keep_archive"] is True
    assert manifest["days"][0]["action"] == "overwrite"


def test_range_manifest_and_request_are_credential_free(tmp_path, range_downloader) -> None:
    def fake_daily(**kwargs):
        return _make_complete_day(range_downloader, kwargs["output_root"], kwargs["requested_date"])

    manifest = range_downloader.run_range(
        start_date="2024-01-01",
        end_date="2024-01-01",
        output_root=tmp_path,
        overwrite=False,
        keep_archive=False,
        continue_on_error=False,
        daily_downloader=fake_daily,
    )
    metadata_dir = range_downloader.range_directory(tmp_path, "2024-01-01", "2024-01-01")
    request = json.loads((metadata_dir / "range_request.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads((metadata_dir / "range_manifest.json").read_text(encoding="utf-8"))
    assert request["times"] == list(range_downloader.HOURLY_TIMES)
    assert saved_manifest["summary"] == manifest["summary"]
    assert all(word not in json.dumps(saved_manifest).lower() for word in ("token", "password", "api_key"))


def test_range_script_does_not_duplicate_daily_cds_schema(range_downloader) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "reanalysis-era5-single-levels" not in source
    assert "10m_u_component_of_wind" not in source
    assert "shell=True" not in source
