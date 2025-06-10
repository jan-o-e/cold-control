import pandas as pd
import numpy as np

# Parameters
input_file = r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\marina\06_06\stokes_optimized_avui.csv'       # Replace with your file path
output_file = r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\marina\normalised\stokes_optimized_avui.csv'     # Replace with your desired output path
rescale_factor = 1           # Adjust this factor as needed

# Step 1: Read CSV with a single row
df = pd.read_csv(input_file, header=None)

# Step 2: Normalize to [0, 1]
values = df.iloc[0].values.astype(float)
normalized = (values - np.min(values)) / (np.max(values) - np.min(values))

# Step 3: Rescale
rescaled = normalized * rescale_factor

# Step 4: Save to CSV
rescaled_df = pd.DataFrame([rescaled])
rescaled_df.to_csv(output_file, index=False, header=False)

print(f"Processed data saved to: {output_file}")