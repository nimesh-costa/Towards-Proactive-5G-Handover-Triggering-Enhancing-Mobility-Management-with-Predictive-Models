import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "5G5G_4G5G_5G4G_DowbSamoling.csv")

df = pd.read_csv(CSV_FILE)

# Handover label counts
counts = df["IsHandover"].value_counts()

print("=" * 50)
print("HANDOVER LABEL INFORMATION")
print("=" * 50)

print("Number of 0s:", (df["IsHandover"] == 0).sum())
print("Number of 1s:", (df["IsHandover"] == 1).sum())

print("\nClass distribution:")
print(counts.sort_index())

print("\nDataset size:", df.shape)