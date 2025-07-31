import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from packing3d import Display
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

DEFAULT_BASE_DIR = Path("/home/henri/prl_cp/tote_consolidation/stats")

console = Console()

def extract_datetime_from_name(name: str) -> datetime:
    # Match the timestamp in the format: YYYY-MM-DD_HH-MM-SS
    match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', name)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    else:
        # If parsing fails, use epoch 0 to sort it last
        return datetime.min

def get_latest_data_dirs(base_dir: Path, limit: int = 10):
    all_dirs = [
        d for d in base_dir.glob("Isaac-Pack-NoArm-v0_*") if d.is_dir()
    ]
    sorted_dirs = sorted(
        all_dirs,
        key=lambda d: extract_datetime_from_name(d.name),
        reverse=True
    )
    return sorted_dirs[:limit]

@click.command()
@click.option("--data_path", type=click.Path(), default=None, help="Path to data directory.")
@click.option("--env_id", required=True, type=int, help="Environment index to process.")
@click.option("--step_num", default=None, type=int, help="Step to visualize.")
def main(data_path, env_id, step_num):
    if not data_path:
        latest_dirs = get_latest_data_dirs(DEFAULT_BASE_DIR)
        table = Table(title="Available Data Directories", show_lines=True)
        table.add_column("Index", justify="center")
        table.add_column("Directory Name", justify="left")

        for i, d in enumerate(latest_dirs):
            table.add_row(str(i), str(d.name))

        console.print(table)

        selection = Prompt.ask("Select a data directory", default="0")
        selected_idx = int(selection)
        data_path = str(latest_dirs[selected_idx])

    console.print(f"Using data path: [bold green]{data_path}[/bold green]")

    container_dir = os.path.join(data_path, "containers", f"env_{env_id}")
    temp_img_dir = os.path.join(data_path, "temp_figures")
    os.makedirs(temp_img_dir, exist_ok=True)

    temp_env_img_dir = os.path.join(temp_img_dir, f"env_{env_id}")
    os.makedirs(temp_env_img_dir, exist_ok=True)

    step_files = sorted(
        [f for f in os.listdir(container_dir) if f.endswith(".pkl")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    step_file = step_files[step_num] if step_num is not None else step_files[-1]

    with open(os.path.join(container_dir, step_file), "rb") as f:
        first_container = pickle.load(f)

    box_size = first_container.box_size
    print("Box size:", box_size)
    display = Display(box_size)
    display.show3d(first_container.geometry)
    plt.savefig("container.png")

    display.show2d(first_container.heightmap)
    plt.savefig("tmp.png")

    # Get box size and heightmap shape
    box_size = first_container.box_size  # e.g., [26, 51, 34]
    heightmap = first_container.heightmap
    heightmap_shape = heightmap.shape  # (H, W)

    # Compute real pixel dimensions in cm
    dx_cm = 1
    dy_cm = 1
    MAX_HEIGHT_CM = box_size[0]

    unused_height = np.clip(MAX_HEIGHT_CM - first_container.heightmap, 0, None)
    # Set all values at MAX_HEIGHT_CM to 0
    unused_height_augmented = unused_height.copy()
    # unused_height_augmented[unused_height == MAX_HEIGHT_CM] = 0
    unused_height_augmented[unused_height < 5] = 0
    display.show2d(unused_height_augmented)
    plt.savefig("unused_height_map_aug.png")

    volume_per_pixel = unused_height * dx_cm * dy_cm
    total_unused_volume_cm3 = np.sum(volume_per_pixel)
    total_unused_volume_L = total_unused_volume_cm3 / 1000.0
    total_used_volume_cm3 = np.sum(first_container.heightmap * dx_cm * dy_cm)
    total_unused_augmented_volume_cm3 = np.sum(unused_height_augmented * dx_cm * dy_cm)
    display.show2d(unused_height)
    plt.savefig("unused_height_map.png")

    console.print(f"[yellow]Unused volume:[/yellow] {total_unused_volume_cm3:.2f} cm³ ({total_unused_volume_L:.2f} L)")
    console.print(f"[yellow]Unused augmented volume:[/yellow] {total_unused_augmented_volume_cm3:.2f} cm³ ({total_unused_augmented_volume_cm3 / 1000.0:.2f} L)")
    console.print(f"[cyan]Used volume:[/cyan] {total_used_volume_cm3:.2f} cm³ ({total_used_volume_cm3 / 1000.0:.2f} L)")
    console.print(f"[green]Total volume:[/green] {total_unused_volume_cm3 + total_used_volume_cm3:.2f} cm³ ({(total_unused_volume_cm3 + total_used_volume_cm3) / 1000.0:.2f} L)")
    console.print(f"[blue]Box volume:[/blue] {np.prod(box_size):.2f} cm³ ({np.prod(box_size) / 1000.0:.2f} L)")

if __name__ == "__main__":
    main()
