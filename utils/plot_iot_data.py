import pandas as pd
import matplotlib.pyplot as plt


def plot_iot_data(df, user_id=None, figsize=(15, 12), save_path="plot.png"):
    """
    Plot IoT sensor data in 6 aligned subplots.

    Args:
        df (pd.DataFrame): DataFrame with columns
            ['timestamp','temperature_c','humidity_pct',
             'fridge_power_w','front_door_open','fire_alarm','anomaly_class']
        user_id (str/int, optional): If given, used in plot title.
        figsize (tuple): Size of the figure.
        save_path (str, optional): If given, save figure to this path.
    """
    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)

    # 1. Temperature
    axes[0].plot(range(len(df["timestamp"])), df["temperature_c"], label="Temperature (°C)", color="tab:red")
    axes[0].set_ylabel("°C")
    axes[0].set_title(f"User {user_id} IoT Data" if user_id else "IoT Data")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Humidity
    axes[1].plot(range(len(df["timestamp"])), df["humidity_pct"], label="Humidity (%)", color="tab:blue")
    axes[1].set_ylabel("%")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Fridge Power
    axes[2].plot(range(len(df["timestamp"])), df["fridge_power_w"], label="Fridge Power (W)", color="tab:green")
    axes[2].set_ylabel("W")
    axes[2].legend()
    axes[2].grid(True)

    # 4. Front Door
    axes[3].step(range(len(df["timestamp"])), df["front_door_open"], label="Front Door Open", color="tab:orange")
    axes[3].set_ylabel("Open=1")
    axes[3].legend()
    axes[3].grid(True)

    # 5. Fire Alarm
    axes[4].step(range(len(df["timestamp"])), df["fire_alarm"], label="Fire Alarm", color="tab:red")
    axes[4].set_ylabel("On=1")
    axes[4].legend()
    axes[4].grid(True)

    # 6. Anomaly Class
    axes[5].step(range(len(df["timestamp"])), df["anomaly_class"], label="Anomaly Class", color="tab:purple")
    axes[5].set_ylabel("Class")
    axes[5].set_xlabel("Timestamp")
    axes[5].legend()
    axes[5].grid(True)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved at {save_path}")

