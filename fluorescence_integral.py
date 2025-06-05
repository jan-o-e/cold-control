import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv(r'C:\Users\apc\Documents\marina\06_jun\04-06\summary_integrals.csv')  # Replace with your actual filename

# Extract imaging power from folder name using regex
df['imaging_power'] = df['folder'].str.extract(r'_(\d{3})_img').astype(float) / 1000

# Sort by imaging power for a cleaner plot
df = df.sort_values('imaging_power')

# Plot
plt.figure(figsize=(8, 5))
plt.plot(df['imaging_power'], df['average_integral'], marker='o')
plt.xlabel('Imaging Beam Power (W)')
plt.ylabel('Average Integral')
plt.title('Fluorescence Integral vs Imaging Beam Power')
plt.grid(True)
plt.tight_layout()
plt.show()
