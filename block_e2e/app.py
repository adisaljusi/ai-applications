import gradio as gr
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ZURICH_HB_LAT = 47.3769
ZURICH_HB_LON = 8.5417

FEATURES = [
    "rooms",
    "area",
    "pop",
    "pop_dens",
    "frg_pct",
    "emp",
    "tax_income",
    "distance_to_zurich_hb",
    "room_per_m2",
    "furnished",
    "zurich_city",
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


with open(SCRIPT_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

df_muni = pd.read_csv(SCRIPT_DIR / "municipality_lookup.csv")

town_choices = sorted(df_muni["bfs_name"].tolist())
town_to_bfs = dict(zip(df_muni["bfs_name"], df_muni["bfs_number"]))


def predict(rooms, area, town, furnished):
    bfs_number = town_to_bfs[town]
    row = df_muni[df_muni["bfs_number"] == bfs_number].iloc[0]

    distance = haversine(row["lat"], row["lon"], ZURICH_HB_LAT, ZURICH_HB_LON)
    room_per_m2 = area / rooms if rooms > 0 else 0
    is_zurich = 1 if row["zurich_city"] == 1 else 0

    X = pd.DataFrame(
        [
            {
                "rooms": rooms,
                "area": area,
                "pop": row["pop"],
                "pop_dens": row["pop_dens"],
                "frg_pct": row["frg_pct"],
                "emp": row["emp"],
                "tax_income": row["tax_income"],
                "distance_to_zurich_hb": distance,
                "room_per_m2": room_per_m2,
                "furnished": int(furnished),
                "zurich_city": is_zurich,
            }
        ]
    )

    prediction = model.predict(X[FEATURES])[0]
    return f"CHF {prediction:,.0f} / month"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Rooms", value=3.5, minimum=1, maximum=10, step=0.5),
        gr.Number(label="Area (m2)", value=80, minimum=10, maximum=500, step=1),
        gr.Dropdown(choices=town_choices, label="Town", value="Zürich"),
        gr.Checkbox(label="Furnished", value=False),
    ],
    outputs=gr.Text(label="Predicted Monthly Rent"),
    title="Zurich Apartment Rent Predictor",
    description="Predict monthly rental prices for apartments in the canton of Zurich. "
    "Select the town, enter apartment details, and get a price estimate.",
    examples=[
        [3.5, 80, "Zürich", False],
        [4.5, 120, "Winterthur", False],
        [2.0, 50, "Dietikon", True],
        [5.0, 150, "Küsnacht (ZH)", False],
    ],
)

if __name__ == "__main__":
    demo.launch()
