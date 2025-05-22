from typing import Optional, Tuple, Union

import mujoco
import numpy as np
from dm_robotics.transformations import transformations as tr


def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    ddx_max: float = 0.0,
) -> np.ndarray:
    # Compute error.
    x_err = x - x_des
    dx_err = dx

    # Apply gains.
    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]

    # Limit maximum error.
    if ddx_max > 0.0:
        x_err_sq_norm = np.sum(x_err**2)
        ddx_max_sq = ddx_max**2
        if x_err_sq_norm > ddx_max_sq:
            x_err *= ddx_max / np.sqrt(x_err_sq_norm)

    return x_err + dx_err


def jointspace(
    model,
    data,
    dof_ids: np.ndarray,
    joint: np.ndarray,
    gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0, 200.0, 150.0, 150.0, 150.0),
    damping_ratio: float = 0.05,
    max_acceleration: Optional[float] = None,
    gravity_comp: bool = True,
) -> np.ndarray:
    
    kp = np.asarray(gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv = np.stack([kp, kd], axis=-1)

    ddx_max = max_acceleration if max_acceleration is not None else 0.0

    joint_pos = np.stack(
        [data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
    ).ravel()
    joint_vel = np.stack(
        [data.sensor(f"panda/joint{i}_vel").data for i in range(1, 8)],
    ).ravel()

    # Compute position PD control.
    tau = pd_control(
        x=joint_pos,
        x_des=joint,
        dx=joint_vel,
        kp_kv=kp_kv,
        ddx_max=ddx_max,
    )

    if gravity_comp:
        tau += data.qfrc_bias[dof_ids]
    return tau
