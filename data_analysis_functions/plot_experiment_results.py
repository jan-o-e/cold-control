import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np

def plot_averaged_shot(shot_folder: str, suffix='_averaged.csv', save=True):
    """
    Plot all iterations in grey and the averaged shot in color.

    Parameters:
        shot_folder (str): Path to the shot folder.
        suffix (str): Filename suffix for the averaged CSV.
        save (bool): Save the figure in the same folder.
    """
    shot_path = Path(shot_folder)
    files = sorted(shot_path.glob("iteration_*_data.csv"))
    avg_file = shot_path / f"{shot_path.name}{suffix}"
    
    # Plot all raw iterations
    plt.figure(figsize=(10, 6))
    for file in files:
        df = pd.read_csv(file)
        plt.plot(df["Time (s)"], df["Channel 4 Voltage (V)"].rolling(window=64).mean(), color='gray', alpha=0.3)

    if suffix == '_aligned.csv':
        time_correction = 0.6e-3
    else:
        time_correction = 0.0

    # Plot the averaged CSV
    if avg_file.exists():
        avg_df = pd.read_csv(avg_file)
        plt.plot(avg_df["Time (s)"]+time_correction, avg_df["Channel 4 Voltage (V)"], color='blue', label=f'{suffix[1:-4]}')

    plt.xlabel("Time (s)")
    plt.ylabel("Channel 4 Voltage (V)")
    plt.title(f"Averaged and Individual Shots: {shot_path.name}")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(shot_path / f"{shot_path.name}_plot{suffix[:-4]}.png")
    plt.close()


def plot_experiment_summary(summary_path: str, save_path: str = None):
    """
    Plot integrated_value vs shot number for each parameter folder.

    Parameters:
        summary_path (str): Path to the experiment_summary.csv.
        save_path (str): Path to save the plot image. If None, doesn't save.
    """
    df = pd.read_csv(summary_path)
    df['shot_number'] = df['shot_folder'].str.extract(r'(\d+)').astype(int)

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    grouped = df.groupby('parameter_folder')

    for name, group in grouped:
        sorted_group = group.sort_values('shot_number')
        plt.errorbar(
            sorted_group['shot_number'],
            sorted_group['integral_value'],
            yerr=sorted_group['integral_uncertainty'],
            fmt='-o', label=name
        )

    plt.xlabel("Shot Number")
    plt.ylabel("Integrated Value (Channel 4)")
    plt.title("Integrated Fluorescence vs Shot Number")
    plt.legend(title="Parameter Folder", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_all_shots_in_folder(root_folder: str, suffix='_averaged.csv'):
    """
    Iterate through all shots in parameter folders and generate individual plots.

    Parameters:
        root_folder (str): Root experiment folder.
        suffix (str): Averaged CSV filename suffix.
    """
    root = Path(root_folder)
    for param_folder in root.iterdir():
        if not param_folder.is_dir():
            continue
        for shot_folder in param_folder.iterdir():
            if shot_folder.is_dir() and shot_folder.name.startswith("shot"):
                try:
                    plot_averaged_shot(str(shot_folder), suffix=suffix)
                except Exception as e:
                    print(f"Failed to plot {shot_folder}: {e}")


if __name__ == "__main__":
    # root = r"d:\pulse_shaping_data\2025-06-24\22-54-51"
    # summary_csv_averaged = os.path.join(root, "experiment_summary_averaged.csv")
    # summary_csv_aligned = os.path.join(root, "experiment_summary_aligned.csv")

    # # Plot experiment summary
    # plot_experiment_summary(summary_csv_averaged, save_path=os.path.join(root, "summary_plot_averaged.png"))
    # plot_experiment_summary(summary_csv_aligned, save_path=os.path.join(root, "summary_plot_aligned.png"))

    # # Plot all shots
    # plot_all_shots_in_folder(root, suffix='_averaged.csv')
    # plot_all_shots_in_folder(root, suffix='_aligned.csv')

    while True:
        user_input = input("Enter the root folder path or 'exit' to quit: ")
        if user_input.lower() in ['exit', "x", "e"]:
            break
        if Path(user_input).is_dir():
            root = Path(user_input)
            summary_csv_averaged = os.path.join(root, "experiment_summary_averaged.csv")
            summary_csv_aligned = os.path.join(root, "experiment_summary_aligned.csv")

            # Plot experiment summary
            print(f"Plotting summaries for {root}...")
            #plot_experiment_summary(summary_csv_averaged, save_path=os.path.join(root, "summary_plot_averaged.png"))
            plot_experiment_summary(summary_csv_aligned, save_path=os.path.join(root, "summary_plot_aligned.png"))
            print("Summaries plotted successfully.")

            # Plot all shots
            print(f"Plotting all shots in {root}...")
            #plot_all_shots_in_folder(root, suffix='_averaged.csv')
            plot_all_shots_in_folder(root, suffix='_aligned.csv')
            print("All shots plotted successfully.")
        else:
            print("Invalid folder path. Please try again.")