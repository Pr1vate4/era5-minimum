from pathlib import Path

import cdsapi


DATASET = "reanalysis-era5-single-levels"

OUTPUT_PATH = Path("data/raw/era5_single_2024_01_01.nc")

REQUEST = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "total_precipitation",
        "total_cloud_cover",
        "total_column_water_vapour",
    ],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": [
        "00:00",
        "06:00",
        "12:00",
        "18:00",
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
}


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Файл уже существует: {OUTPUT_PATH}")
        print("Удалите его вручную, если нужно скачать данные заново.")
        return

    print("Отправляем запрос в Climate Data Store...")
    print(f"Файл будет сохранён в: {OUTPUT_PATH}")

    client = cdsapi.Client()

    result = client.retrieve(
        DATASET,
        REQUEST,
    )

    result.download(str(OUTPUT_PATH))

    size_mb = OUTPUT_PATH.stat().st_size / 1024**2

    print("Скачивание завершено.")
    print(f"Файл: {OUTPUT_PATH}")
    print(f"Размер: {size_mb:.2f} МБ")


if __name__ == "__main__":
    main()
