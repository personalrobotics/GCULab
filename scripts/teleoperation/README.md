
# Data collection
## Recording demos
with Keyboard
```bash
python scripts/teleoperation/pack_so3.py --task=Isaac-Pack-NoArm-Camera-v0 --num_envs 1 --enable_cameras  --seed 0 --dataset_file ./datasets/dataset.hdf5
```
or with Mello
```bash
python scripts/teleop_se3_agent.py --task Isaac-Pack-UR5-Teleop-v0 --num_envs 1 --teleop_device mello
```

## Replaying demos
```bash
python scripts/teleoperation/replay_demos.py --task Isaac-Pack-NoArm-Camera-v0 --device cpu --dataset_file ./datasets/dataset.hdf5 --enable_cameras --seed 0
```

Recorded demos will be saved in the HDF5 file specified in `--dataset_file`. Data recorded is structured as follows

| **Field** | **Description** | **Shape / Format** | **Notes** |
|--------------------|-----------------|--------------------|-----------|
| **actions** | Recorded teleoperation actions | `(9)` | `tote_id, obj_id, x, y, z, quat_x, quat_y, quat_z, quat_w` |
| **initial_state/** | Initial configuration of all rigid objects in the environment before simulation begins | — | Contains one group per object (`object0`, `object1`, ..., `objectN`) |
| ├─ `root_pose` | 3D position and quaternion rotation | `(N, 7)` | `(x, y, z, w, x, y, z)` |
| └─ `root_velocity` | Linear (3) + angular (3) velocity | `(N, 6)` | — |
| **pre_step_actions/** | Raw policy outputs before being processed into actions | — | Captures unnormalized policy outputs |
| **pre_step_flat_policy_observations/** | Policy observations | — | Typically `obj_dim` or `obj_latent` vectors |
| **pre_step_flat_sensor_observations/** | Sensor data before each step | — | Usually heightmaps or visual observations |
| **post_step_processed_actions/** | Processed actions after normalization | — | Derived from `pre_step_actions/` |
| **post_step_states/** | Object states after each simulation step | Same as `initial_state/` | Recorded at every timestep |
| **Compatibility** | Works directly with **robomimic** algorithms | — | HDF5 can be visualized at [myhdf5.hdfgroup.org](https://myhdf5.hdfgroup.org/) or loaded in Python |
