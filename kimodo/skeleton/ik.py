# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inverse-kinematics primitives: analytic two-bone IK with pole vector and FABRIK fallback."""

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


def _orthogonalize(R: torch.Tensor) -> torch.Tensor:
    """Project a 3x3 matrix back onto SO(3) via SVD."""
    U, _, Vt = torch.linalg.svd(R)
    # Ensure proper rotation (det = +1)
    det = torch.det(U @ Vt)
    sign = torch.diag(torch.tensor([1.0, 1.0, det.sign()], device=R.device, dtype=R.dtype))
    return U @ sign @ Vt


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
    v_from = v_from.to(dtype=torch.float64) / (v_from.norm() + 1e-12)
    v_to = v_to.to(dtype=torch.float64) / (v_to.norm() + 1e-12)

    cross = torch.cross(v_from, v_to, dim=0)
    sin_angle = cross.norm()
    cos_angle = torch.dot(v_from, v_to).clamp(-1.0, 1.0)

    if sin_angle < 1e-6:
        if cos_angle > 0:
            return torch.eye(3, device=v_from.device, dtype=v_from.dtype)
        # ~180-degree rotation: pick an arbitrary perpendicular axis
        perp = torch.tensor([1.0, 0.0, 0.0], device=v_from.device, dtype=v_from.dtype)
        if torch.abs(torch.dot(v_from, perp)) > 0.9:
            perp = torch.tensor([0.0, 1.0, 0.0], device=v_from.device, dtype=v_from.dtype)
        axis = torch.cross(v_from, perp, dim=0)
        axis = axis / (axis.norm() + 1e-12)
        return 2.0 * torch.outer(axis, axis) - torch.eye(3, device=v_from.device, dtype=v_from.dtype)

    axis = cross / sin_angle
    K = torch.zeros(3, 3, device=v_from.device, dtype=v_from.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    R = torch.eye(3, device=v_from.device, dtype=v_from.dtype) + sin_angle * K + (1.0 - cos_angle) * (K @ K)
    return R


def _get_rest_bone_lengths(skeleton, chain_indices: List[int], device, dtype) -> torch.Tensor:
    """Compute bone lengths from the rest pose (stable, never drifts)."""
    neutral = skeleton.neutral_joints.to(device=device, dtype=dtype)
    bone_lengths = torch.zeros(len(chain_indices) - 1, device=device, dtype=dtype)
    for i in range(len(chain_indices) - 1):
        bone_lengths[i] = (neutral[chain_indices[i + 1]] - neutral[chain_indices[i]]).norm()
    return bone_lengths


def two_bone_ik_solve(
    chain_positions: torch.Tensor,
    target_position: torch.Tensor,
    pole_position: torch.Tensor,
    bone_lengths: torch.Tensor,
) -> torch.Tensor:
    """Analytic two-bone IK solver with pole vector control.

    Solves for positions of 3 joints (root, mid, end) such that the end
    effector reaches the target and the mid joint bends toward the pole.

    Args:
        chain_positions: Current joint positions, shape ``(3, 3)``.
        target_position: Desired end-effector position, shape ``(3,)``.
        pole_position: Pole vector target controlling bend plane, shape ``(3,)``.
        bone_lengths: Upper and lower bone lengths, shape ``(2,)``.

    Returns:
        Solved joint positions, shape ``(3, 3)``.
    """
    root = chain_positions[0]
    L1 = bone_lengths[0]
    L2 = bone_lengths[1]

    positions = chain_positions.clone()

    to_target = target_position - root
    D_raw = to_target.norm()
    max_reach = L1 + L2

    target_dir = to_target / (D_raw + 1e-12)

    if D_raw >= max_reach - 1e-6:
        # Fully extended toward target
        positions[1] = root + target_dir * L1
        positions[2] = root + target_dir * max_reach
        return positions

    min_reach = torch.abs(L1 - L2)
    D = D_raw.clamp(min_reach + 1e-6, max_reach - 1e-6)

    # Law of cosines: angle at root joint
    cos_a = (L1 * L1 + D * D - L2 * L2) / (2.0 * L1 * D)
    cos_a = cos_a.clamp(-1.0, 1.0)
    sin_a = torch.sqrt((1.0 - cos_a * cos_a).clamp(min=0.0))

    # Project pole onto plane perpendicular to target direction
    to_pole = pole_position - root
    pole_proj = to_pole - torch.dot(to_pole, target_dir) * target_dir
    pole_norm = pole_proj.norm()

    if pole_norm < 1e-6:
        # Pole is colinear with root→target, pick a perpendicular fallback
        perp = torch.tensor([1.0, 0.0, 0.0], device=root.device, dtype=root.dtype)
        if torch.abs(torch.dot(target_dir, perp)) > 0.9:
            perp = torch.tensor([0.0, 1.0, 0.0], device=root.device, dtype=root.dtype)
        pole_dir = torch.cross(target_dir, perp, dim=0)
        pole_dir = pole_dir / (pole_dir.norm() + 1e-12)
    else:
        pole_dir = pole_proj / pole_norm

    # Mid joint: rotated from root→target axis by the solved angle
    positions[1] = root + target_dir * (L1 * cos_a) + pole_dir * (L1 * sin_a)
    # End effector at clamped target distance
    positions[2] = root + target_dir * D

    return positions


def compute_default_pole_position(
    root_pos: torch.Tensor,
    mid_pos: torch.Tensor,
    end_pos: torch.Tensor,
    offset: float = 0.5,
) -> torch.Tensor:
    """Compute a default pole vector position from current joint positions.

    Projects the mid joint's bend direction outward to create a sensible
    default pole target.

    Args:
        root_pos: Root joint position, shape ``(3,)``.
        mid_pos: Mid joint position, shape ``(3,)``.
        end_pos: End joint position, shape ``(3,)``.
        offset: Distance to push the pole from the mid joint.

    Returns:
        Default pole position, shape ``(3,)``.
    """
    root_to_end = end_pos - root_pos
    root_to_end_len = root_to_end.norm()
    if root_to_end_len < 1e-8:
        return mid_pos.clone()
    root_to_end_dir = root_to_end / root_to_end_len

    root_to_mid = mid_pos - root_pos
    proj_len = torch.dot(root_to_mid, root_to_end_dir)
    proj_point = root_pos + root_to_end_dir * proj_len

    bend_vec = mid_pos - proj_point
    bend_norm = bend_vec.norm()

    if bend_norm < 1e-6:
        # Chain is nearly straight, choose a perpendicular direction
        perp = torch.tensor([0.0, 0.0, 1.0], device=mid_pos.device, dtype=mid_pos.dtype)
        if torch.abs(torch.dot(root_to_end_dir, perp)) > 0.9:
            perp = torch.tensor([0.0, 1.0, 0.0], device=mid_pos.device, dtype=mid_pos.dtype)
        bend_dir = perp - torch.dot(perp, root_to_end_dir) * root_to_end_dir
        bend_dir = bend_dir / (bend_dir.norm() + 1e-12)
    else:
        bend_dir = bend_vec / bend_norm

    return mid_pos + bend_dir * offset


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
    skeleton,
    current_local_rots: torch.Tensor,
    root_pos: torch.Tensor,
) -> torch.Tensor:
    """Recover local rotations consistent with FABRIK-solved positions.

    Instead of incrementally swinging existing (potentially drifted) global
    rotations, this computes each chain joint's local rotation from scratch
    by running FK up the chain and computing the rotation that maps the
    rest-pose bone direction to the solved bone direction in the parent's frame.
    Twist is preserved from the original local rotations.

    Args:
        chain_indices: Joint indices from chain root to EE.
        solved_positions: Solved chain positions, shape ``(len(chain_indices), 3)``.
        skeleton: Skeleton instance.
        current_local_rots: Current local rotations, shape ``(J, 3, 3)``.
        root_pos: Current root world position, shape ``(3,)``.

    Returns:
        Updated local rotations for the full skeleton, shape ``(J, 3, 3)``.
    """
    device = current_local_rots.device
    dtype = current_local_rots.dtype
    new_local_rots = current_local_rots.clone()
    neutral = skeleton.neutral_joints.to(device=device, dtype=dtype)

    # Build global rotations incrementally via FK for chain joints only.
    # We need the parent's world rotation to convert bone directions to local frame.
    # Start by computing global rots for the full skeleton from current local rots
    # (this gives us the parent rotation for the chain root).
    _, global_rots_current = batch_rigid_transform(
        current_local_rots[None],
        neutral[None],
        skeleton.joint_parents.to(device),
        skeleton.root_idx,
    )
    global_rots_current = global_rots_current[0]

    # Track the evolving global rotation as we walk down the chain
    chain_global_rots = {}

    for ci in range(len(chain_indices) - 1):
        joint_idx = chain_indices[ci]
        child_idx = chain_indices[ci + 1]
        parent_idx = int(skeleton.joint_parents[joint_idx].item())

        # Get parent's world rotation
        if parent_idx < 0:
            R_parent_world = torch.eye(3, device=device, dtype=dtype)
        elif parent_idx in chain_global_rots:
            R_parent_world = chain_global_rots[parent_idx]
        else:
            R_parent_world = global_rots_current[parent_idx]

        # Rest-pose bone vector (in local frame of parent)
        rest_bone = neutral[child_idx] - neutral[joint_idx]
        if rest_bone.norm() < 1e-10:
            chain_global_rots[joint_idx] = R_parent_world @ new_local_rots[joint_idx]
            continue

        # Target bone direction in world space from FABRIK
        target_bone_world = solved_positions[ci + 1] - solved_positions[ci]
        if target_bone_world.norm() < 1e-10:
            chain_global_rots[joint_idx] = R_parent_world @ new_local_rots[joint_idx]
            continue

        # Current bone direction in world space (from current rotation, not positions)
        current_bone_world = global_rots_current[joint_idx] @ rest_bone
        current_bone_world_dir = current_bone_world / (current_bone_world.norm() + 1e-12)
        target_bone_world_dir = target_bone_world / (target_bone_world.norm() + 1e-12)

        # Swing rotation in world space (computed in float64 for stability)
        R_swing = _rotation_from_to(current_bone_world_dir, target_bone_world_dir).to(dtype=dtype)

        # New global rotation for this joint
        new_global_rot = R_swing @ global_rots_current[joint_idx]

        # Orthogonalize to prevent drift
        new_global_rot = _orthogonalize(new_global_rot)

        # Convert to local: L = R_parent^T @ R_global
        new_local_rot = R_parent_world.T @ new_global_rot
        new_local_rots[joint_idx] = new_local_rot

        # Store for children in the chain
        chain_global_rots[joint_idx] = new_global_rot

    return new_local_rots


def solve_ik_chain(
    skeleton,
    joints_pos: torch.Tensor,
    joints_rot: torch.Tensor,
    joints_local_rot: torch.Tensor,
    chain_indices: List[int],
    target_pos: torch.Tensor,
    pole_pos: Optional[torch.Tensor] = None,
    tolerance: float = 1e-4,
    max_iterations: int = 20,
) -> tuple:
    """Solve IK for a single chain and return updated full-skeleton pose data.

    Uses analytic two-bone IK with pole vector for 3-joint chains, or falls
    back to FABRIK for longer chains.

    Args:
        skeleton: Skeleton instance.
        joints_pos: Current global joint positions, shape ``(J, 3)``.
        joints_rot: Current global joint rotations, shape ``(J, 3, 3)``.
        joints_local_rot: Current local joint rotations, shape ``(J, 3, 3)``.
        chain_indices: Ordered joint indices from chain root to EE.
        target_pos: Desired EE world position, shape ``(3,)``.
        pole_pos: Pole vector position for two-bone IK, shape ``(3,)``.
            Required for 3-joint chains; ignored for longer chains.
        tolerance: FABRIK convergence tolerance (FABRIK fallback only).
        max_iterations: Maximum FABRIK iterations (FABRIK fallback only).

    Returns:
        Tuple ``(new_joints_pos, new_joints_local_rot, new_joints_rot)`` with
        the same shapes as the inputs.
    """
    device = joints_pos.device
    dtype = joints_pos.dtype

    # Extract chain positions
    chain_positions = joints_pos[chain_indices].clone()

    # Use rest-pose bone lengths (stable, prevents drift from accumulating)
    bone_lengths = _get_rest_bone_lengths(skeleton, chain_indices, device, dtype)

    # Use two-bone IK for 3-joint chains when a pole vector is provided
    if len(chain_indices) == 3 and pole_pos is not None:
        solved_positions = two_bone_ik_solve(
            chain_positions,
            target_pos.to(device=device, dtype=dtype),
            pole_pos.to(device=device, dtype=dtype),
            bone_lengths,
        )
    else:
        solved_positions = fabrik_solve(
            chain_positions, target_pos.to(device=device, dtype=dtype),
            bone_lengths, tolerance, max_iterations,
        )

    # Recover local rotations from solved positions (drift-resistant)
    new_local_rots = _recover_rotations(
        chain_indices, solved_positions, skeleton, joints_local_rot,
        joints_pos[skeleton.root_idx].clone(),
    )

    # Run FK to get consistent global state
    root_pos = joints_pos[skeleton.root_idx].clone()
    neutral = skeleton.neutral_joints.to(device=device, dtype=dtype)
    pelvis_offset = neutral[skeleton.root_idx]

    new_posed_joints, new_global_rots_full = batch_rigid_transform(
        new_local_rots[None],
        neutral[None],
        skeleton.joint_parents.to(device),
        skeleton.root_idx,
    )
    new_posed_joints = new_posed_joints[0] + root_pos[None] - pelvis_offset[None]
    new_global_rots_full = new_global_rots_full[0]

    return new_posed_joints, new_local_rots, new_global_rots_full
