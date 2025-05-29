import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_mean_pd_signal(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)

    # Check for required columns
    required_columns = {'Time (s)', 'Mean PD Signal', 'Readout Index'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    # Get unique readout indices
    readout_indices = sorted(df['Readout Index'].unique())

    # Create the plot
    plt.figure(figsize=(10, 6))

    for rid in readout_indices:
        subset = df[df['Readout Index'] == rid]
        plt.plot(subset['Time (s)'], subset['Mean PD Signal'], label=f'Readout {rid}')

    # Format plot
    plt.title('Mean PD Signal Over Time by Readout Index')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean PD Signal')
    plt.legend(title='Readout Index')
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()

# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_mean_pd_signal.py path_to_csv_file")
    else:
        plot_mean_pd_signal(sys.argv[1])

