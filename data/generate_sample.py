# # # data/generate_sample.py
# # import pandas as pd
# # import numpy as np
# # import os

# # def generate(n=400, seed=42):
# #     np.random.seed(seed)
# #     # Simulate three ecosystem patch clusters with different energy-flow patterns
# #     c1 = np.random.normal(loc=[3.5, 50, 0.12, 15], scale=[0.6, 10, 0.02, 2], size=(int(n/3), 4))
# #     c2 = np.random.normal(loc=[6.0, 80, 0.08, 25], scale=[0.7, 12, 0.015, 3], size=(int(n/3), 4))
# #     c3 = np.random.normal(loc=[1.5, 30, 0.18, 10], scale=[0.4, 8, 0.02, 1.5], size=(n - 2*int(n/3), 4))
# #     data = np.vstack([c1, c2, c3])
# #     df = pd.DataFrame(data, columns=[
# #         "NPP",                 # net primary productivity (proxy)
# #         "biomass",             # biomass
# #         "trophic_efficiency",  # trophic transfer efficiency
# #         "temperature"          # local temperature
# #     ])
# #     # Add some noise and missing values
# #     for _ in range(int(0.02*n)):
# #         i = np.random.randint(0, df.shape[0])
# #         j = np.random.randint(0, df.shape[1])
# #         df.iat[i, j] = np.nan
# #     # Add coordinates (optional) just for mapping later
# #     df["lat"] = np.random.uniform(8.0, 37.0, size=df.shape[0])
# #     df["lon"] = np.random.uniform(68.0, 97.0, size=df.shape[0])
# #     # Save
# #     os.makedirs("data", exist_ok=True)
# #     df.to_csv("data/sample_data.csv", index=False)
# #     print("Saved data/sample_data.csv")

# # if __name__ == "__main__":
# #     generate()


# import pandas as pd
# import numpy as np

# # Configure size
# n_rows = 50000  # for example 50k rows

# np.random.seed(42)
# data = {
#     "Feature1": np.random.normal(50, 15, n_rows),
#     "Feature2": np.random.normal(100, 25, n_rows),
#     "Feature3": np.random.normal(0, 1, n_rows),
#     "Feature4": np.random.uniform(0, 500, n_rows),
#     "Feature5": np.random.exponential(1.0, n_rows),
#     "lat": np.random.uniform(-90, 90, n_rows),
#     "lon": np.random.uniform(-180, 180, n_rows)
# }

# df = pd.DataFrame(data)

# # Introduce some missing values randomly
# for col in df.columns:
#     df.loc[df.sample(frac=0.05, random_state=42).index, col] = np.nan

# df.to_csv("data/large_synthetic_numeric.csv", index=False)
# print("✅ large_synthetic_numeric.csv created, rows:", n_rows)



import pandas as pd
import numpy as np
import os

# ---------- Configuration ----------
n_rows = 10000   # 👈 change this number to any size (e.g., 10000 or 50000)

np.random.seed(42)

# ---------- Generate Synthetic Data ----------
data = {
    "NPP": np.random.normal(50, 15, n_rows),          # Net Primary Productivity
    "Biomass": np.random.normal(100, 25, n_rows),     # Biomass
    "Temperature": np.random.uniform(5, 35, n_rows),  # Environmental temperature
    "Trophic_Efficiency": np.random.uniform(0, 1, n_rows),  # Trophic level efficiency
    "Energy_Flow": np.random.exponential(10.0, n_rows),     # Random energy flow
    "lat": np.random.uniform(-90, 90, n_rows),
    "lon": np.random.uniform(-180, 180, n_rows)
}

df = pd.DataFrame(data)

# ---------- Add Some Missing Values ----------
for col in df.columns:
    df.loc[df.sample(frac=0.05, random_state=42).index, col] = np.nan

# ---------- Save to CSV ----------
os.makedirs("data", exist_ok=True)
file_path = os.path.join("data", "large_sample_data.csv")
df.to_csv(file_path, index=False)

print(f"✅ Dataset created: {file_path}")
print(f"Rows: {n_rows}, Columns: {len(df.columns)}")
