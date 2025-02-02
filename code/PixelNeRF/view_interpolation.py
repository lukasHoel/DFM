import torch
from scipy.spatial.transform import Rotation as R


@torch.no_grad()
def interpolate_pose(
    initial, final, t: float,
):
    # Get the relative rotation.
    r_initial = initial[:3, :3]
    r_final = final[:3, :3]
    r_relative = r_final @ r_initial.T

    r_relative = r_relative.float()

    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=final.dtype, device=final.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial[:3, 3]
    t_final = final[:3, 3]
    t_interpolated = t_initial + (t_final - t_initial) * t

    # Assemble the result.
    result = torch.zeros_like(initial)
    result[3, 3] = 1
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result


@torch.no_grad()
def interpolate_intrinsics(
    initial, final, t: float,
):
    return initial + (final - initial) * t
