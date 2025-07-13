import numpy as np
from packing3d import Item, PackingProblem, Display
import matplotlib.pyplot as plt
import torch

# One env only for now
scale = 1.0

def create_packing_problem(tote_manager, packable_objects):
    """
    Create a packing problem using the tote manager.
    """
    tote_dims, obj_dims, obj_voxels = get_packing_variables(tote_manager)
    items = []
    for i in range(len(obj_dims)):
        if i in packable_objects:
            # Create an item for each packable object
            # items.append(getSurfaceItem(obj_dims[i][0], obj_dims[i][1], obj_dims[i][2]))
            items.append(Item(np.array(obj_voxels[0][i], dtype=np.float32)))

    # x y z to z x y 
    display = Display(tote_dims)

    return PackingProblem(tote_dims, items), tote_dims, items, display

def get_packing_variables(tote_manager):
    tote_dims = tote_manager.true_tote_dim.tolist()
    tote_dims = [int(tote_dims[2] * scale), int(tote_dims[0] * scale), int(tote_dims[1] * scale)]

    obj_dims = tote_manager.obj_bboxes.tolist()[0] # Only first environment for now
    obj_dims = [[int(dim * scale) for dim in dims] for dims in obj_dims]

    obj_voxels = tote_manager.obj_voxels
    # TODO (kaikwan): Scale voxel grid as well
    return tote_dims, obj_dims, obj_voxels


def getSurfaceItem(xSize, ySize, zSize):

    cube = np.ones((xSize, ySize, zSize))
    cube[1: xSize-1, 1: ySize-1, 1: zSize-1] = 0

    return Item(cube)

def preview_packing_problem(tote_manager, packable_objects):
    """
    Preview the packing problem in 3D.
    """
    packable_objects = packable_objects[0]  # Only first environment for now
    problem, tote_dims, items = create_packing_problem(tote_manager, packable_objects)
    display = Display(tote_dims)
    
    for idx in range(len(items)):
        curr_item: Item = items[idx]
        transforms = problem.container.search_possible_position(curr_item, step_width=90)
        assert len(transforms) > 0, f"No valid position found for item {idx}."
        print("Transforms found for item {}: {}".format(idx, transforms[0]))

        curr_item.transform(transforms[0])

        problem.container.add_item(curr_item)
        display.show3d(problem.container.geometry)
        plt.savefig(f"packing_problem_{idx}.png")

def get_action(problem, items, obj_indicies, display):
    """
    Get the action for the current step in the packing problem.
    """
    obj_idx = obj_indicies[torch.randint(0, len(obj_indicies), (1,))][0]

    curr_item = items[obj_idx.item()]
    obj_indicies = obj_idx.unsqueeze(0)
    transforms = problem.container.search_possible_position(curr_item, step_width=90)
    if transforms:
        transform = transforms[0]
        curr_item.transform(transforms[0])
        problem.container.add_item(curr_item)
        display.show3d(problem.container.geometry)
        plt.pause(1.0)
        return problem, transform, obj_indicies
    return None  # No valid position found