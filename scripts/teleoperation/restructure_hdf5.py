#!/usr/bin/env python3
"""
Script to restructure HDF5 dataset from flat to nested format for robomimic compatibility.

This script converts observation data from:
    data/demo_X/obs (flat dataset)
    data/demo_X/obs_sensor (flat dataset)

To the expected robomimic format:
    data/demo_X/obs/obs (nested dataset)
    data/demo_X/obs/obs_sensor (nested dataset)

Additionally adds required metadata attributes like 'env_args' and 'num_samples'.
"""

import argparse
import h5py
import json
import os
import shutil


def restructure_hdf5(input_path: str, output_path: str = None, backup: bool = True, env_name: str = None):
    """
    Restructure HDF5 file to be compatible with robomimic.
    
    Args:
        input_path: Path to the input HDF5 file
        output_path: Path to save the restructured file (default: overwrites input)
        backup: Whether to create a backup of the original file
        env_name: Name of the environment (if None, will try to extract from path or use default)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create backup if requested
    if backup and output_path is None:
        backup_path = input_path.replace('.hdf5', '_backup.hdf5')
        if not os.path.exists(backup_path):
            shutil.copy2(input_path, backup_path)
            print(f"Created backup: {backup_path}")
    
    # If no output path specified, use a temporary file
    if output_path is None:
        output_path = input_path.replace('.hdf5', '_temp.hdf5')
        will_replace = True
    else:
        will_replace = False
    
    print(f"\nRestructuring {input_path}...")
    print(f"Output will be saved to: {output_path}")
    
    # Determine environment name
    if env_name is None:
        # Try to extract from existing metadata or use a default
        with h5py.File(input_path, 'r') as f_check:
            if 'data' in f_check and 'env' in f_check['data'].attrs:
                env_name = f_check['data'].attrs['env']
            else:
                # Default environment name - try to infer from path or use generic
                env_name = "Isaac-Pack-NoArm-Camera-v0"  # Default for this project
                print(f"Warning: No env name found, using default: {env_name}")
    
    print(f"Environment name: {env_name}")
    
    # Open input file in read mode
    with h5py.File(input_path, 'r') as f_in:
        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            # Copy metadata from root
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value
            
            # Process each demo
            if 'data' not in f_in:
                raise ValueError("No 'data' group found in input file")
            
            data_group = f_out.create_group('data')
            demo_keys = list(f_in['data'].keys())
            print(f"Found {len(demo_keys)} demonstrations")
            
            # Add environment metadata to data group
            env_meta = {
                "env_name": env_name,
                "type": 2,  # EnvType for IsaacLab/Gym environments
                "env_kwargs": {}
            }
            data_group.attrs["env_args"] = json.dumps(env_meta, indent=4)
            
            # Track total samples
            total_samples = 0
            
            for demo_key in demo_keys:
                print(f"  Processing {demo_key}...", end=' ')
                demo_in = f_in[f'data/{demo_key}']
                demo_out = data_group.create_group(demo_key)
                
                # Copy demo attributes
                for attr_name, attr_value in demo_in.attrs.items():
                    demo_out.attrs[attr_name] = attr_value
                
                # Identify observation keys (anything starting with 'obs' but not in a group)
                obs_keys = []
                non_obs_keys = []
                num_samples = 0
                
                for key in demo_in.keys():
                    if isinstance(demo_in[key], h5py.Dataset):
                        # This is a flat dataset
                        if key.startswith('obs'):
                            obs_keys.append(key)
                            if num_samples == 0:
                                num_samples = demo_in[key].shape[0]
                        else:
                            non_obs_keys.append(key)
                            # Get num_samples from actions if available
                            if key == 'actions' and num_samples == 0:
                                num_samples = demo_in[key].shape[0]
                    else:
                        # This is a group (like 'states', 'initial_state')
                        non_obs_keys.append(key)
                
                # Add num_samples attribute to demo
                if num_samples > 0:
                    demo_out.attrs["num_samples"] = num_samples
                    total_samples += num_samples
                
                # Create nested obs group if we have obs keys
                if obs_keys:
                    obs_group = demo_out.create_group('obs')
                    for obs_key in obs_keys:
                        # Copy dataset from flat structure to nested
                        data = demo_in[obs_key][()]
                        obs_group.create_dataset(obs_key, data=data, dtype=demo_in[obs_key].dtype)
                        print(f"\n    Moved {obs_key} (shape={data.shape}) to obs/{obs_key}", end='')
                
                # Copy non-obs keys as-is
                for key in non_obs_keys:
                    if isinstance(demo_in[key], h5py.Dataset):
                        # Copy dataset
                        data = demo_in[key][()]
                        demo_out.create_dataset(key, data=data, dtype=demo_in[key].dtype)
                    else:
                        # Recursively copy group
                        def copy_group(src_group, dst_group):
                            for k in src_group.keys():
                                if isinstance(src_group[k], h5py.Dataset):
                                    data = src_group[k][()]
                                    dst_group.create_dataset(k, data=data, dtype=src_group[k].dtype)
                                else:
                                    new_group = dst_group.create_group(k)
                                    # Copy attributes
                                    for attr_name, attr_value in src_group[k].attrs.items():
                                        new_group.attrs[attr_name] = attr_value
                                    copy_group(src_group[k], new_group)
                        
                        new_group = demo_out.create_group(key)
                        # Copy attributes
                        for attr_name, attr_value in demo_in[key].attrs.items():
                            new_group.attrs[attr_name] = attr_value
                        copy_group(demo_in[key], new_group)
                
                print(" ✓")
            
            # Add total samples to data group
            data_group.attrs["total"] = total_samples
            print(f"\nTotal samples across all demos: {total_samples}")
    
    # If we created a temp file, replace the original
    if will_replace:
        os.remove(input_path)
        os.rename(output_path, input_path)
        print(f"\n✓ Successfully restructured {input_path}")
    else:
        print(f"\n✓ Successfully created {output_path}")
    
    # Verify the structure
    print("\nVerifying output structure...")
    with h5py.File(input_path if will_replace else output_path, 'r') as f:
        demo_key = list(f['data'].keys())[0]
        demo = f[f'data/{demo_key}']
        print(f"\nSample demo ({demo_key}) structure:")
        for key in demo.keys():
            if isinstance(demo[key], h5py.Group):
                print(f"  {key}/ (Group)")
                for subkey in demo[key].keys():
                    if isinstance(demo[key][subkey], h5py.Dataset):
                        print(f"    {subkey}: Dataset, shape={demo[key][subkey].shape}")
                    else:
                        print(f"    {subkey}/ (Group)")
            else:
                print(f"  {key}: Dataset, shape={demo[key].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Restructure HDF5 dataset for robomimic compatibility"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input HDF5 file to restructure"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the restructured file (default: overwrites input)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup of the original file"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="Environment name (default: Isaac-Pack-NoArm-Camera-v0)"
    )
    
    args = parser.parse_args()
    
    restructure_hdf5(
        args.input_file,
        output_path=args.output,
        backup=not args.no_backup,
        env_name=args.env_name
    )


if __name__ == "__main__":
    main()
