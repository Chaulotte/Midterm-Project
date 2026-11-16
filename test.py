import requests

URL = "http://localhost:9696/predict"
HEALTH_URL = "http://localhost:9696/health"

sample_penguins = [
    {
        "species": "Adelie",
        "island": "Torgersen",
        "culmen_length_mm": 39.1,
        "culmen_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
    },
    {
        "species": "Gentoo",
        "island": "Biscoe",
        "culmen_length_mm": 45.2,
        "culmen_depth_mm": 14.8,
        "flipper_length_mm": 212.0,
        "body_mass_g": 5200.0,
    },
]

if __name__ == "__main__":
    health_resp = requests.get(HEALTH_URL, timeout=5)
    print("Health check:", health_resp.status_code, health_resp.json())

    for idx, penguin in enumerate(sample_penguins, start=1):
        response = requests.post(URL, json=penguin, timeout=5)
        print(f"Sample {idx} status:", response.status_code)
        print(f"Sample {idx} result:", response.json())
