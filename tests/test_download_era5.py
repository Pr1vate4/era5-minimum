"""Offline tests for the safe ERA5 CDS downloader."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_era5.py"


@pytest.fixture(scope="module")
def downloader():
    spec = importlib.util.spec_from_file_location("era5_download_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_netcdf_pair(directory: Path) -> tuple[Path, Path]:
    coords = {
        "valid_time": np.array(["2024-01-01T00:00:00"], dtype="datetime64[ns]"),
        "latitude": [90.0],
        "longitude": [0.0],
    }
    instant = xr.Dataset(
        {
            name: (("valid_time", "latitude", "longitude"), np.ones((1, 1, 1)), {"GRIB_stepType": "instant"})
            for name in ("u10", "v10", "t2m", "msl", "sst", "tcc", "tcwv")
        },
        coords=coords,
    )
    accumulated = xr.Dataset(
        {"tp": (("valid_time", "latitude", "longitude"), np.ones((1, 1, 1)), {"GRIB_stepType": "accum"})},
        coords=coords,
    )
    instant_path = directory / "anything-instant.nc"
    accumulated_path = directory / "anything-accum.nc"
    instant.to_netcdf(instant_path)
    accumulated.to_netcdf(accumulated_path)
    return instant_path, accumulated_path


def test_request_has_stable_variables_and_times(downloader) -> None:
    request = downloader.build_request("2024-01-01", ["18:00", "00:00", "06:00", "06:00"])
    assert request["variable"] == list(downloader.REQUEST_VARIABLES)
    assert request["time"] == ["00:00", "06:00", "18:00"]
    assert request["year"] == ["2024"]
    assert request["data_format"] == "netcdf"


@pytest.mark.parametrize("value", ["2024-02-30", "01-01-2024", "2024-1-01"])
def test_invalid_date_is_rejected(downloader, value: str) -> None:
    with pytest.raises(ValueError, match="Invalid --date"):
        downloader.build_request(value, ["00:00"])


@pytest.mark.parametrize("value", ["24:00", "06:60", "6:00"])
def test_invalid_time_is_rejected(downloader, value: str) -> None:
    with pytest.raises(ValueError, match="Invalid --times"):
        downloader.build_request("2024-01-01", [value])


def test_dry_run_is_offline_and_creates_no_directory(tmp_path, downloader, capsys, monkeypatch) -> None:
    output_root = tmp_path / "raw"
    monkeypatch.delitem(sys.modules, "cdsapi", raising=False)
    result = downloader.main(
        ["--date", "2024-01-01", "--times", "00:00", "06:00", "--output-root", str(output_root), "--dry-run"]
    )
    assert result == 0
    assert not output_root.exists()
    assert "cdsapi" not in sys.modules
    assert "DRY RUN" in capsys.readouterr().out


def test_existing_target_is_not_overwritten_without_flag(tmp_path, downloader) -> None:
    target = downloader.dataset_directory(tmp_path, "2024-01-01")
    target.mkdir(parents=True)
    result = downloader.main(["--date", "2024-01-01", "--times", "00:00", "--output-root", str(tmp_path)])
    assert result == 2
    assert target.is_dir()


def test_zip_detection_and_safe_extraction(tmp_path, downloader) -> None:
    archive = tmp_path / "download.part"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/file.txt", "safe")
    assert downloader.detect_download_format(archive) == "zip"
    extracted = downloader.safe_extract_zip(archive, tmp_path / "extracted")
    assert [path.relative_to(tmp_path / "extracted").as_posix() for path in extracted] == ["nested/file.txt"]


def test_zip_path_traversal_is_rejected(tmp_path, downloader) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "no")
    with pytest.raises(downloader.DownloadError, match="Unsafe ZIP member path"):
        downloader.safe_extract_zip(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside.txt").exists()


def test_discovers_pair_from_netcdf_metadata(tmp_path, downloader) -> None:
    _write_netcdf_pair(tmp_path)
    pair = downloader.discover_netcdf_pair(tmp_path)
    assert set(pair) == {"instant", "accumulated"}


def test_prepares_zip_into_confirmed_raw_pair_without_tp6h(tmp_path, downloader) -> None:
    source = tmp_path / "source"
    source.mkdir()
    instant, accumulated = _write_netcdf_pair(source)
    archive = tmp_path / "download.part"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(instant, "nested/instant-file.nc")
        bundle.write(accumulated, "nested/accum-file.nc")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "request.json").write_text("{}\n", encoding="utf-8")

    archive_detected, extracted, hashes = downloader._prepare_dataset(prepared, archive, keep_archive=True)

    assert archive_detected is True
    assert extracted == [downloader.INSTANT_FILENAME, downloader.ACCUMULATED_FILENAME]
    assert (prepared / downloader.INSTANT_FILENAME).is_file()
    assert (prepared / downloader.ACCUMULATED_FILENAME).is_file()
    assert (prepared / "source_download.zip").is_file()
    assert "tp6h" not in " ".join(path.name for path in prepared.iterdir())
    assert set(hashes) == {downloader.INSTANT_FILENAME, downloader.ACCUMULATED_FILENAME, "request.json"}


def test_checksum_manifest_is_sha256sum_compatible(tmp_path, downloader) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    hashes = downloader.write_checksum_manifest(tmp_path, ["one.txt", "two.txt"])
    assert hashes["one.txt"] == downloader.sha256_file(tmp_path / "one.txt")
    assert (tmp_path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()[0].endswith("  one.txt")


def test_metadata_has_no_credentials_and_no_tp6h(tmp_path, downloader) -> None:
    metadata = downloader.build_metadata(
        requested_date="2024-01-01",
        requested_times=["00:00"],
        target=tmp_path / "era5_single_2024_01_01",
        archive_detected=True,
        extracted_files=[downloader.INSTANT_FILENAME, downloader.ACCUMULATED_FILENAME],
        file_sizes={downloader.INSTANT_FILENAME: 1, downloader.ACCUMULATED_FILENAME: 2},
        hashes={downloader.INSTANT_FILENAME: "a", downloader.ACCUMULATED_FILENAME: "b"},
    )
    encoded = json.dumps(metadata).lower()
    assert "token" not in encoded and "password" not in encoded and "api_key" not in encoded
    assert "tp6h" not in metadata["requested_variables"]
    assert "not automatically tp6h" in metadata["warnings"][0]


def test_cdsapi_import_is_lazy(downloader, monkeypatch) -> None:
    calls: list[str] = []

    def missing(name: str):
        calls.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(downloader.importlib, "import_module", missing)
    with pytest.raises(downloader.DownloadError, match="pip install -e"):
        downloader._cds_client()
    assert calls == ["cdsapi"]


@pytest.mark.parametrize("wrapper", ["download_era5_day.py", "download_era5_example.py"])
def test_deprecated_wrappers_delegate_to_main_downloader(tmp_path, wrapper: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT.parent / wrapper),
            "--date",
            "2024-01-01",
            "--times",
            "00:00",
            "--output-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "DEPRECATION" in result.stderr
    assert "DRY RUN" in result.stdout
