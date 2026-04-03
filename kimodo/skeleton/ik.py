# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inverse-kinematics primitives using the FABRIK algorithm."""

from typing import List, Optional

import torch

from .kinematics import batch_rigid_transform
from .transforms import global_rots_to_local_rots


def get_chain_indices(joint_parents: torch.Tensor, ee_idx: int, chain_root_idx: int) -> List[int]:
    """Walk from *ee_idx* up through *joint_parents* to *chain_root_idx*.

    Args:
        joint_parents: Parent-index tensor of shape ``(J,)`` with root parent ``-1``.
        ee_idx: Index of the end-effector joint.
        chain_root_idx: Index of the chain root joint (inclusive).

    Returns:
        Joint indices ordered from chain root to end-effector (inclusive).

    Raises:
        ValueError: If *chain_root_idx* is not an ancestor of *ee_idx*.
    """
    chain = [ee_idx]
    current = ee_idx
    while current != chain_root_idx:
        parent = int(joint_parents[current].item())
        if parent == -1:
            raise ValueError(
                f"chain_root_idx {chain_root_idx} is not an ancestor of ee_idx {ee_idx}"
            )
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain


def _rotation_from_to(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    """Compute the minimal rotation matrix mapping unit vector *v_from* to *v_to*.

    Uses the Rodrigues formula via the cross-product.  When the vectors are
    (anti-)parallel the result degrades gracefully.

    Args:
        v_from: Unit vector of shape ``(3,)``.
        v_to: Unit vector of shape ``(3,)``.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    v_from = v_from / (v_from.norm() + 1e-12)
    v_to = v_to / (v_to.norm() + 1e-12)

    cross = torch.cross(v_from, v_to, dim=0)
    sin_angle = cross.norm()
    cos_angle = torch.dot(v_from, v_to)

    if sin_angle < 1e-8:
        if cos_angle > 0:
            return torch.eye(3, device=v_from.device, dtype=v_from.dtype)
        # ~180-degree rotation: pick an arbitrary perpendicular axis
        perp = torch.tensor([1.0, 0.0, 0.0], device=v_from.device, dtype=v_from.dtype)
        if torch.abs(torch.dot(v_from, perp)) > 0.9:
            perp = torch.tensor([0.0, 1.0, 0.0], device=v_from.device, dtype=v_from.dtype)
        axis = torch.cross(v_from, perp, dim=0)
        axis = axis / (axis.norm() + 1e-12)
        # 180-degree rotation around axis: R = 2 * outer(axis, axis) - I
        return 2.0 * torch.outer(axis, axis) - torch.eye(3, device=v_from.device, dtype=v_from.dtype)

    axis = cross / sin_angle
    # Rodrigues: R = I + sin(a)*K + (1-cos(a))*K^2, where K is the skew-symmetric of axis
    K = torch.zeros(3, 3, device=v_from.device, dtype=v_from.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    return torch.eye(3, device=v_from.device, dtype=v_from.dtype) + sin_angle * K + (1.0 - cos_angle) * (K @ K)


def fabrik_solve(
    chain_positions: torch.Tensor,
    target_position: torch.Tensor,
    bone_lengths: torch.Tensor,
    tolerance: float = 1e-4,
    max_iterations: int = 20,
) -> torch.Tensor:
    """Run the FABRIK algorithm to solve a positional IK chain.

    Args:
        chain_positions: Current joint positions along the chain, shape ``(N, 3)``,
            ordered from chain root to end-effector.
        target_position: Desired world-space position for the end-effector, shape ``(3,)``.
        bone_lengths: Bone lengths for each segment, shape ``(N-1,)``.
        tolerance: Stop iterating when the EE is within this distance of the target.
        max_iterations: Maximum number of forward/backward passes.

    Returns:
        Solved joint positions with shape ``(N, 3)``.
    """
    N = chain_positions.shape[0]
    positions = chain_positions.clone()
    root_pos = positions[0].clone()

    total_reach = bone_lengths.sum()
    dist_to_target = (target_position - root_pos).norm()

    # If the target is unreachable, extend the chain fully toward the target
    if dist_to_target > total_reach:
        direction = (target_position - root_pos)
        direction = direction / (direction.norm() + 1e-12)
        for i in range(1, N):
            positions[i] = positions[i - 1] + direction * bone_lengths[i - 1]
        return positions

    for _iteration in range(max_iterations):
        # Check convergence
        ee_error = (positions[-1] - target_position).norm()
        if ee_error < tolerance:
            break

        # --- Backward pass: move EE to target, walk toward root ---
        positions[-1] = target_position.clone()
        for i in range(N - 2, -1, -1):
            diff = positions[i] - positions[i + 1]
            diff_len = diff.norm() + 1e-12
            positions[i] = positions[i + 1] + (diff / diff_len) * bone_lengths[i]

        # --- Forward pass: fix root, walk toward EE ---
        positions[0] = root_pos
        for i in range(1, N):
            diff = positions[i] - positions[i - 1]
            diff_len = diff.norm() + 1e-12
            positions[i] = positions[i - 1] + (diff / diff_len) * bone_lengths[i - 1]

    return positions


def _recover_rotations(
    chain_indices: List[int],
    solved_positions: torch.Tensor,
    all_joints_pos: torch.Tensor,
    all_global_rots: torch.Tensor,
    skeleton,
) -> torch.Tensor:
    """Recover global rotation matrices consistent with FABRIK-solved positions.

    For each bone in the chain, computes the swing rotation that maps the old
    world-space bone direction to the solved bone direction, preserving the
    original twist around the bone axis.

    Args:
        chain_indices: Joint indices from chain root to EE (as returned by
            :func:`get_chain_indices`).
        solved_positions: Solved chain positions from :func:`fabrik_solve`,
            shape ``(len(chain_indices), 3)``.
        all_joints_pos: Full skeleton positions at the current frame, shape ``(J, 3)``.
        all_global_rots: Full skeleton global rotations, shape ``(J, 3, 3)``.
        skeleton: Skeleton instance with ``neutral_joints`` and ``joint_parents``.

    Returns:
        Updated global rotations for the full skeleton, shape ``(J, 3, 3)``.
    """
    device = all_global_rots.device
    dtype = all_global_rots.dtype
    new_global_rots = all_global_rots.clone()

    neutral = skeleton.neutral_joints.to(device=device, dtype=dtype)

    # Process each joint in the chain except the EE (last one)
    # because the EE's rotation doesn't affect any bone direction in this chain
    for ci in range(len(chain_indices) - 1):
        joint_idx = chain_indices[ci]
        child_idx = chain_indices[ci + 1]

        # Rest-pose bone vector (in skeleton local frame)
        rest_bone = neutral[child_idx] - neutral[joint_idx]
        rest_bone_len = rest_bone.norm()
        if rest_bone_len < 1e-10:
            continue

        # Old world-space bone direction
        old_world_dir = (all_joints_pos[child_idx] - all_joints_pos[joint_idx])
        old_world_dir = old_world_dir / (old_world_dir.norm() + 1e-12)

        # New world-space bone direction from FABRIK solution
        new_world_dir = (solved_positions[ci + 1] - solved_positions[ci])
        new_world_dir = new_world_dir / (new_world_dir.norm() + 1e-12)

        # Swing rotation from old to new direction
        R_swing = _rotation_from_to(old_world_dir, new_world_dir)

        # Apply swing to preserve twist
        new_global_rots[joint_idx] = R_swing @ new_global_rots[joint_idx]

    return new_global_rots


def solve_ik_chain(
    skeleton,
    joints_pos: torch.Tensor,
    joints_rot: torch.Tensor,
    joints_local_rot: torch.Tensor,
    chain_indices: List[int],
    target_pos: torch.Tensor,
    tolerance: float = 1e-4,
    max_iterations: int = 20,
) -> tuple:
    """Solve IK for a single chain and return updated full-skeleton pose data.

    Runs FABRIK to find solved positions, recovers rotation matrices that are
    consistent with those positions (preserving twist), and re-runs FK to
    obtain a globally consistent skeleton state.

    Args:
        skeleton: Skeleton instance.
        joints_pos: Current global joint positions, shape ``(J, 3)``.
        joints_rot: Current global joint rotations, shape ``(J, 3, 3)``.
        joints_local_rot: Current local joint rotations, shape ``(J, 3, 3)``.
        chain_indices: Ordered joint indices from chain root to EE.
        target_pos: Desired EE world position, shape ``(3,)``.
        tolerance: FABRIK convergence tolerance.
        max_iterations: Maximum FABRIK iterations.

    Returns:
        Tuple ``(new_joints_pos, new_joints_local_rot, new_joints_rot)`` with
        the same shapes as the inputs.
    """
    device = joints_pos.device
    dtype = joints_pos.dtype

    # Extract chain positions
    chain_positions = joints_pos[chain_indices].clone()

    # Compute bone lengths from current positions (not rest pose, to handle scaling)
    bone_lengths = torch.zeros(len(chain_indices) - 1, device=device, dtype=dtype)
    for i in range(len(chain_indices) - 1):
        bone_lengths[i] = (chain_positions[i + 1] - chain_positions[i]).norm()

    # Solve positions with FABRIK
    solved_positions = fabrik_solve(
        chain_positions, target_pos.to(device=device, dtype=dtype),
        bone_lengths, tolerance, max_iterations,
    )

    # Recover global rotations from solved positions
    new_global_rots = _recover_rotations(
        chain_indices, solved_positions, joints_pos, joints_rot, skeleton,
    )

    # Convert to local rotations
    new_local_rots = global_rots_to_local_rots(new_global_rots[None], skeleton)[0]

    # Copy non-chain joints from original local rotations
    new_local_rots_full = joints_local_rot.clone()
    for idx in chain_indices[:-1]:  # EE rotation unchanged
        new_local_rots_full[idx] = new_local_rots[idx]

    # Run FK to get consistent global state
    root_pos = joints_pos[skeleton.root_idx].clone()
    new_posed_joints, new_global_rots_full = batch_rigid_transform(
        new_local_rots_full[None],
        skeleton.neutral_joints[None].to(device=device, dtype=dtype),
        skeleton.joint_parents.to(device),
        skeleton.root_idx,
    )
    # Add root position offset
    pelvis_offset = skeleton.neutral_joints[skeleton.root_idx].to(device=device, dtype=dtype)
    new_posed_joints = new_posed_joints[0] + root_pos[None] - pelvis_offset[None]
    new_global_rots_full = new_global_rots_full[0]

    return new_posed_joints, new_local_rots_full, new_global_rots_full
