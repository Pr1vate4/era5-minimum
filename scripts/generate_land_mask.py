import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def project_point(
    longitude: float,
    latitude: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Переводит долготу и широту в координаты изображения."""
    x = (longitude + 180.0) / 360.0 * width
    y = (90.0 - latitude) / 180.0 * height
    return x, y


def draw_ring(
    draw: ImageDraw.ImageDraw,
    ring: list[list[float]],
    width: int,
    height: int,
    fill: int,
) -> None:
    points = [
        project_point(longitude, latitude, width, height)
        for longitude, latitude, *_ in ring
    ]

    if len(points) >= 3:
        draw.polygon(points, fill=fill)


def draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: list[list[list[float]]],
    width: int,
    height: int,
) -> None:
    if not polygon:
        return

    # Первый контур — сама суша.
    draw_ring(draw, polygon[0], width, height, fill=255)

    # Остальные контуры — отверстия, например внутренние озёра.
    for hole in polygon[1:]:
        draw_ring(draw, hole, width, height, fill=0)


def generate_mask(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    supersampling: int,
) -> None:
    with input_path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    render_width = width * supersampling
    render_height = height * supersampling

    image = Image.new(
        mode="L",
        size=(render_width, render_height),
        color=0,
    )
    draw = ImageDraw.Draw(image)

    for feature in geojson.get("features", []):
        geometry = feature.get("geometry")

        if not geometry:
            continue

        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        if geometry_type == "Polygon":
            draw_polygon(
                draw,
                coordinates,
                render_width,
                render_height,
            )

        elif geometry_type == "MultiPolygon":
            for polygon in coordinates:
                draw_polygon(
                    draw,
                    polygon,
                    render_width,
                    render_height,
                )

    if supersampling > 1:
        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")

    print(f"Маска создана: {output_path}")
    print(f"Размер: {width} × {height}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Создание чёрно-белой маски суши из GeoJSON.",
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--supersampling", type=int, default=2)

    args = parser.parse_args()

    generate_mask(
        input_path=Path(args.input),
        output_path=Path(args.output),
        width=args.width,
        height=args.height,
        supersampling=args.supersampling,
    )


if __name__ == "__main__":
    main()