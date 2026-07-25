from fastapi.testclient import TestClient

from era5_minimum.api.app import app
from era5_minimum.api.weather import WeatherProviderError, get_weather_provider


class Provider:
    def dataset_metadata(self): return {"id": "real", "is_mock": False}
    def variables(self): return [{"logical_name": "t2m", "kind": "surface"}, {"logical_name": "T850", "kind": "pressure", "default_level": 850}]
    def timestamps(self): return ["2020-01-01T00:00:00Z"]
    def layer(self, variable, timestamp, level, target_width, target_height):
        if variable not in {"t2m", "T850"}: raise WeatherProviderError("unknown variable")
        if variable == "T850" and level != 850: raise WeatherProviderError("T850 requires level=850")
        return {"variable": variable, "timestamp": timestamp, "level": level, "shape": [target_height, target_width], "is_mock": False}


def setup_function(): app.dependency_overrides[get_weather_provider] = lambda: Provider()
def teardown_function(): app.dependency_overrides.clear()


def test_weather_metadata_variables_and_timestamps():
    client = TestClient(app)
    assert client.get("/api/v1/datasets/current").json()["is_mock"] is False
    assert len(client.get("/api/v1/variables").json()) == 2
    assert client.get("/api/v1/timestamps").json() == ["2020-01-01T00:00:00Z"]


def test_weather_layer_modes_and_validation():
    client = TestClient(app)
    assert client.get("/api/v1/layers?variable=t2m&timestamp=2020-01-01T00:00:00Z&target_width=8&target_height=4").json()["shape"] == [4, 8]
    assert client.get("/api/v1/layers?variable=T850&timestamp=2020-01-01T00:00:00Z&level=850").status_code == 200
    assert client.get("/api/v1/layers?variable=T850&timestamp=2020-01-01T00:00:00Z").status_code == 422
    assert client.get("/api/v1/layers?variable=t2m&timestamp=2020-01-01T00:00:00Z&mode=reconstructed").status_code == 501
