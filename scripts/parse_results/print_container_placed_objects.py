#!/usr/bin/env python3

import argparse
import os
import pickle
from pathlib import Path

obj_bbox = {
    "003_cracker_box.usd": (22, 17, 8),
    "004_sugar_box.usd": (18, 10, 5),
    "036_wood_block.usd": (21, 10, 9),
}

def main():
    parser = argparse.ArgumentParser(description="Print the asset path filenames of placed objects in a container")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to the container pickle file")
    parser.add_argument("--show_positions", action="store_true", help="Also display position information")
    parser.add_argument("--show_attitudes", action="store_true", help="Also display attitude information")
    parser.add_argument("--show_bbox", action="store_true", help="Also display bounding box information")
    parser.add_argument("--bbox_only", action="store_true", help="Display only filename and bounding box information in a compact format")

    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        print(f"Error: File {args.pkl_path} does not exist")
        return

    # Load the container from the pickle file
    try:
        with open(args.pkl_path, "rb") as f:
            container = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # Check if the container has placed_items
    if not hasattr(container, "placed_items"):
        print("Container does not have a 'placed_items' attribute")
        return

    # Print information about the placed items
    if not container.placed_items:
        print("No items placed in the container")
        return

    if args.bbox_only:
        print(f"Container has {len(container.placed_items)} placed items:")
        for i, item in enumerate(container.placed_items, 1):
            filename = os.path.basename(item.asset_path)
            bbox = obj_bbox.get(filename, (0, 0, 0))
            print(f"{list(bbox)}, ")
        return

    print(f"Container has {len(container.placed_items)} placed items:\n")

    for i, item in enumerate(container.placed_items, 1):
        asset_path = item.asset_path
        filename = os.path.basename(asset_path)

        output = f"{i}. {filename}"

        if args.show_positions:
            pos = item.position
            output += f" at position (x={pos.x}, y={pos.y}, z={pos.z})"

        if args.show_attitudes:
            att = item.attitude
            output += f" with attitude (roll={att.roll:.2f}, pitch={att.pitch:.2f}, yaw={att.yaw:.2f})"

        if args.show_bbox:
            bbox = obj_bbox.get(filename, (0, 0, 0))
            output += f" with bounding box {bbox}"

        print(output)

        # Print the full asset path below each item
        print(f"   Full path: {asset_path}\n")


if __name__ == "__main__":
    main()
