import pandas as pd

df = pd.read_csv("nisqa_results/NISQA_results.csv")
df_fake = df[df["deg"].str.startswith("fake_")]
print(f"Mean NISQA: {df_fake['mos_pred'].mean():.3f}")
print(f"Median: {df_fake['mos_pred'].median():.3f}")
print(f"Std: {df_fake['mos_pred'].std():.3f}")
