import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
from packing3d import Display
from PIL import Image


def process_step(container_dir, temp_env_img_dir, step_file, box_size=None):
    step = os.path.splitext(step_file)[0]
    img_path = os.path.join(temp_env_img_dir, f"step_{step}.png")

    if os.path.exists(img_path):
        print(f"Image for step {step} already exists. Skipping.")
        return

    pkl_path = os.path.join(container_dir, step_file)
    with open(pkl_path, "rb") as f:
        container = pickle.load(f)

    display = Display(box_size) if box_size is not None else Display(container.box_size)
    display.show3d(container.geometry)

    plt.savefig(img_path)
    plt.clf()
    print(f"Saved figure for step {step} as {img_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D container packing visualizations and GIFs.")
    parser.add_argument("--data_path", type=str, required=True, help="Base path where the container data is saved.")
    parser.add_argument("--env_id", type=int, required=True, help="Environment index to process.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to process (optional).")

    args = parser.parse_args()

    container_dir = os.path.join(args.data_path, "containers", f"env_{args.env_id}")
    temp_img_dir = os.path.join(args.data_path, "temp_figures")
    os.makedirs(temp_img_dir, exist_ok=True)

    temp_env_img_dir = os.path.join(temp_img_dir, f"env_{args.env_id}")
    os.makedirs(temp_env_img_dir, exist_ok=True)

    step_files = sorted(
        [f for f in os.listdir(container_dir) if f.endswith(".pkl")], key=lambda x: int(os.path.splitext(x)[0])
    )

    if args.max_steps is not None:
        step_files = step_files[: args.max_steps]

    with open(os.path.join(container_dir, step_files[0]), "rb") as f:
        first_container = pickle.load(f)
    box_size = first_container.box_size

    with ProcessPoolExecutor() as executor:
        futures = []
        for step_file in step_files:
            futures.append(executor.submit(process_step, container_dir, temp_env_img_dir, step_file, box_size))

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print("Error in processing:", e)

    img_files = sorted(
        [f for f in os.listdir(temp_env_img_dir) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    if args.max_steps is not None:
        img_files = img_files[: args.max_steps]

    images = []
    for img_file in img_files:
        img_path = os.path.join(temp_env_img_dir, img_file)
        img = Image.open(img_path).convert("RGBA")
        images.append(img)

    if images:
        gif_path = os.path.join(temp_env_img_dir, f"animation_env_{args.env_id}.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
        print(f"Saved animated GIF: {gif_path}")
    else:
        print("No images found to create GIF.")


if __name__ == "__main__":
    main()
