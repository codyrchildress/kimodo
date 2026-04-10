# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FBX import utilities for loading animation data into Kimodo motion dicts.

Requires the Autodesk FBX Python SDK (``fbx`` module).  The SDK is not
pip-installable — download from https://aps.autodesk.com/developer/overview/fbx-sdk
and copy the Python bindings into your site-packages.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _import_fbx():
    """Import the Autodesk FBX SDK Python module with a helpful error."""
    try:
        import fbx as fbx_sdk

        return fbx_sdk
    except ImportError:
        raise ImportError(
            "FBX import requires the Autodesk FBX Python SDK ('fbx' module).\n"
            "Download from: https://aps.autodesk.com/developer/overview/fbx-sdk\n"
            "After installing the SDK, copy the Python bindings to your Python site-packages.\n"
            "Alternatively, convert your FBX to BVH using Blender (File > Export > BVH)."
        )


# ---------------------------------------------------------------------------
# Joint name mapping
# ---------------------------------------------------------------------------

# Mixamo names use a "mixamorig:" prefix and slightly different leg naming.
_MIXAMO_TO_SOMA77 = {
    "Hips": "Hips",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Chest",
    "Neck": "Neck1",
    "Head": "Head",
    "HeadTop_End": "HeadEnd",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "LeftHandThumb1": "LeftHandThumb1",
    "LeftHandThumb2": "LeftHandThumb2",
    "LeftHandThumb3": "LeftHandThumb3",
    "LeftHandIndex1": "LeftHandIndex1",
    "LeftHandIndex2": "LeftHandIndex2",
    "LeftHandIndex3": "LeftHandIndex3",
    "LeftHandMiddle1": "LeftHandMiddle1",
    "LeftHandMiddle2": "LeftHandMiddle2",
    "LeftHandMiddle3": "LeftHandMiddle3",
    "LeftHandRing1": "LeftHandRing1",
    "LeftHandRing2": "LeftHandRing2",
    "LeftHandRing3": "LeftHandRing3",
    "LeftHandPinky1": "LeftHandPinky1",
    "LeftHandPinky2": "LeftHandPinky2",
    "LeftHandPinky3": "LeftHandPinky3",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "RightHandThumb1": "RightHandThumb1",
    "RightHandThumb2": "RightHandThumb2",
    "RightHandThumb3": "RightHandThumb3",
    "RightHandIndex1": "RightHandIndex1",
    "RightHandIndex2": "RightHandIndex2",
    "RightHandIndex3": "RightHandIndex3",
    "RightHandMiddle1": "RightHandMiddle1",
    "RightHandMiddle2": "RightHandMiddle2",
    "RightHandMiddle3": "RightHandMiddle3",
    "RightHandRing1": "RightHandRing1",
    "RightHandRing2": "RightHandRing2",
    "RightHandRing3": "RightHandRing3",
    "RightHandPinky1": "RightHandPinky1",
    "RightHandPinky2": "RightHandPinky2",
    "RightHandPinky3": "RightHandPinky3",
    "LeftUpLeg": "LeftLeg",
    "LeftLeg": "LeftShin",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "LeftToe_End": "LeftToeEnd",
    "RightUpLeg": "RightLeg",
    "RightLeg": "RightShin",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
    "RightToe_End": "RightToeEnd",
}


def _strip_namespace(name: str) -> str:
    """Strip namespace prefixes like ``mixamorig:`` or ``Bip01_``."""
    # mixamorig:Hips -> Hips
    if ":" in name:
        name = name.rsplit(":", 1)[-1]
    # Bip01_Spine -> Spine (common 3ds Max prefix)
    for prefix in ("Bip01_", "Bip01 ", "Bip02_", "Bip02 "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def _build_joint_name_mapping(
    fbx_names: list[str],
    target_names: list[str],
) -> dict[str, int]:
    """Map FBX joint names to target skeleton joint indices.

    Tries direct match, namespace-stripped match, Mixamo alias, and
    case-insensitive fallback.
    """
    target_idx = {n: i for i, n in enumerate(target_names)}
    target_lower = {n.lower(): i for i, n in enumerate(target_names)}
    mapping: dict[str, int] = {}

    for fbx_name in fbx_names:
        # 1. Direct match
        if fbx_name in target_idx:
            mapping[fbx_name] = target_idx[fbx_name]
            continue

        stripped = _strip_namespace(fbx_name)

        # 2. Stripped direct match
        if stripped in target_idx:
            mapping[fbx_name] = target_idx[stripped]
            continue

        # 3. Mixamo alias
        aliased = _MIXAMO_TO_SOMA77.get(stripped)
        if aliased and aliased in target_idx:
            mapping[fbx_name] = target_idx[aliased]
            continue

        # 4. Case-insensitive
        if stripped.lower() in target_lower:
            mapping[fbx_name] = target_lower[stripped.lower()]
            continue

    return mapping


# ---------------------------------------------------------------------------
# FBX scene helpers
# ---------------------------------------------------------------------------


def _collect_skeleton_nodes(fbx_sdk, node, joints: list, names: list) -> None:
    """Recursively collect skeleton / null joint nodes from the FBX scene."""
    attr = node.GetNodeAttribute()
    if attr is not None:
        attr_type = attr.GetAttributeType()
        if attr_type in (
            fbx_sdk.FbxNodeAttribute.eSkeleton,
            fbx_sdk.FbxNodeAttribute.eNull,
        ):
            joints.append(node)
            names.append(node.GetName())
    for i in range(node.GetChildCount()):
        _collect_skeleton_nodes(fbx_sdk, node.GetChild(i), joints, names)


def _get_first_anim_stack(fbx_sdk, scene):
    """Return the first animation stack in the scene, or None."""
    anim_stack = scene.GetCurrentAnimationStack()
    if anim_stack is not None:
        return anim_stack
    for i in range(scene.GetSrcObjectCount()):
        obj = scene.GetSrcObject(i)
        if obj.GetClassId() == fbx_sdk.FbxAnimStack.ClassId:
            return obj
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fbx_to_kimodo_motion(
    path: Union[str, Path],
    skeleton=None,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Load an FBX animation file into a Kimodo motion dict.

    The FBX skeleton hierarchy is mapped to the target Kimodo skeleton by
    joint name.  Joints without a match use identity rotations.

    Args:
        path: Path to FBX file.
        skeleton: Target Kimodo skeleton.  Defaults to SOMA77 (77 joints).

    Returns:
        ``(motion_dict, source_fps)`` where ``source_fps`` is the native FBX
        frame rate.
    """
    fbx_sdk = _import_fbx()
    from kimodo.exports.motion_io import complete_motion_dict
    from kimodo.skeleton.registry import build_skeleton

    if skeleton is None:
        skeleton = build_skeleton(77)
    device = skeleton.neutral_joints.device
    path = str(path)

    # ---- initialise SDK ----
    manager = fbx_sdk.FbxManager.Create()
    ios = fbx_sdk.FbxIOSettings.Create(manager, fbx_sdk.IOSROOT)
    manager.SetIOSettings(ios)

    # ---- import scene ----
    importer = fbx_sdk.FbxImporter.Create(manager, "")
    if not importer.Initialize(path, -1, ios):
        status = importer.GetStatus()
        manager.Destroy()
        raise ValueError(f"Failed to load FBX file: {path}\n{status.GetErrorString()}")

    scene = fbx_sdk.FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()

    # ---- normalise coordinate system to Y-up, meters ----
    fbx_sdk.FbxAxisSystem.MayaYUp.ConvertScene(scene)
    scene_unit = scene.GetGlobalSettings().GetSystemUnit()
    scale_to_meters = scene_unit.GetConversionFactorTo(fbx_sdk.FbxSystemUnit.m)

    # ---- animation info ----
    anim_stack = _get_first_anim_stack(fbx_sdk, scene)
    if anim_stack is None:
        manager.Destroy()
        raise ValueError("No animation found in FBX file")

    time_span = anim_stack.GetLocalTimeSpan()
    start_time = time_span.GetStart()
    stop_time = time_span.GetStop()

    time_mode = scene.GetGlobalSettings().GetTimeMode()
    fps = fbx_sdk.FbxTime.GetFrameRate(time_mode)
    if fps <= 0:
        fps = 30.0

    duration = stop_time.GetSecondDouble() - start_time.GetSecondDouble()
    num_frames = max(1, int(round(duration * fps)) + 1)

    # ---- collect skeleton joints ----
    fbx_joints: list = []
    fbx_joint_names: list[str] = []
    _collect_skeleton_nodes(fbx_sdk, scene.GetRootNode(), fbx_joints, fbx_joint_names)

    if not fbx_joints:
        manager.Destroy()
        raise ValueError("No skeleton joints found in FBX file")

    # ---- joint name mapping ----
    target_names = list(skeleton.bone_order_names)
    name_map = _build_joint_name_mapping(fbx_joint_names, target_names)

    matched = len(name_map)
    total_target = len(target_names)
    if matched == 0:
        manager.Destroy()
        raise ValueError(
            f"Could not map any FBX joints to the target skeleton ({skeleton.name}).\n"
            f"FBX joints: {fbx_joint_names[:20]}{'...' if len(fbx_joint_names) > 20 else ''}\n"
            f"Target joints: {target_names[:20]}{'...' if len(target_names) > 20 else ''}"
        )
    if matched < total_target:
        warnings.warn(
            f"Mapped {matched}/{total_target} joints from FBX to {skeleton.name}. "
            f"Unmatched joints will use identity rotations.",
            UserWarning,
            stacklevel=2,
        )

    # ---- sample animation ----
    num_joints = skeleton.nbjoints
    root_kimodo_idx = int(skeleton.root_idx)

    local_rot_mats = (
        torch.eye(3, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(num_frames, num_joints, 3, 3)
        .clone()
    )
    root_positions = torch.zeros(num_frames, 3, device=device, dtype=torch.float32)

    for frame in range(num_frames):
        t = fbx_sdk.FbxTime()
        t.SetSecondDouble(start_time.GetSecondDouble() + frame / fps)

        for fbx_idx, node in enumerate(fbx_joints):
            kimodo_idx = name_map.get(fbx_joint_names[fbx_idx])
            if kimodo_idx is None:
                continue

            local_xform = node.EvaluateLocalTransform(t)

            # Rotation (FBX returns Euler XYZ in degrees)
            rot = local_xform.GetR()
            euler_deg = np.array([rot[0], rot[1], rot[2]], dtype=np.float64)
            rot_mat = Rotation.from_euler("XYZ", np.deg2rad(euler_deg)).as_matrix()
            local_rot_mats[frame, kimodo_idx] = torch.tensor(
                rot_mat, device=device, dtype=torch.float32
            )

            # Root translation (apply unit conversion)
            if kimodo_idx == root_kimodo_idx:
                trans = local_xform.GetT()
                root_positions[frame] = torch.tensor(
                    [
                        trans[0] * scale_to_meters,
                        trans[1] * scale_to_meters,
                        trans[2] * scale_to_meters,
                    ],
                    device=device,
                    dtype=torch.float32,
                )

    manager.Destroy()

    motion_dict = complete_motion_dict(local_rot_mats, root_positions, skeleton, float(fps))
    return motion_dict, float(fps)
