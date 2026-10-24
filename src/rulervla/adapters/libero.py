"""Same-state raw LIBERO to the frozen V0.3 stored sample convention."""
import numpy as np

from rulervla.data.schema import Camera, observation_index, validate_sample
from rulervla.geometry.action_coord import raster, validate_contract


class LiberoAdapter:
    def __init__(self, geometry_contract):
        validate_contract(geometry_contract)
        self.contract = geometry_contract

    def from_observation(self, observation, *, depth_m, intrinsic, camera_to_world,
                         state8, instruction):
        """depth_m is already converted metric optical Z in raw simulator raster.

        Conversion of the normalized MuJoCo depth buffer belongs to the simulator
        caller. Language and wrist camera are carried through unchanged.
        """
        sample = dict(
            agentview_rgb=raster(np.asarray(observation["agentview_image"]), "rot180"),
            wrist_rgb=raster(np.asarray(observation["robot0_eye_in_hand_image"]), "rot180"),
            agentview_depth_m=raster(np.asarray(depth_m), "rot180"),
            intrinsic=np.asarray(intrinsic).copy(),
            camera_to_world=np.asarray(camera_to_world).copy(),
            ee_world_m=np.asarray(observation["robot0_eef_pos"]).copy(),
            eef_quaternion_xyzw=np.asarray(observation["robot0_eef_quat"]).copy(),
            state8=np.asarray(state8, dtype=np.float32).copy(), task=instruction,
        )
        validate_sample(sample)
        return sample

    def from_metric_observation(self, observation, *, intrinsic, camera_to_world,
                                wrist_intrinsic, wrist_camera_to_world, state8, instruction,
                                episode_index, control_step, calibration_step,
                                depth_m=None, wrist_depth_m=None):
        """两路图像、状态和相机标定必须来自同一个控制步；深度和四元数可选。"""
        episode_index = observation_index(episode_index)
        control_step = observation_index(control_step)
        calibration_step = observation_index(calibration_step)
        if calibration_step != control_step:
            raise ValueError('Images, state and both camera calibrations must use the same control step')
        sample = dict(
            agentview_rgb=raster(np.asarray(observation["agentview_image"]), "rot180"),
            wrist_rgb=raster(np.asarray(observation["robot0_eye_in_hand_image"]), "rot180"),
            intrinsic=np.asarray(intrinsic).copy(),
            camera_to_world=np.asarray(camera_to_world).copy(),
            wrist_intrinsic=np.asarray(wrist_intrinsic).copy(),
            wrist_camera_to_world=np.asarray(wrist_camera_to_world).copy(),
            ee_world_m=np.asarray(observation["robot0_eef_pos"]).copy(),
            state8=np.asarray(state8, dtype=np.float32).copy(), task=instruction,
            episode_index=episode_index, frame_index=control_step,
        )
        for key, depth in (("agentview_depth_m", depth_m), ("wrist_depth_m", wrist_depth_m)):
            if depth is not None:
                sample[key] = raster(np.asarray(depth), "rot180")
        if "robot0_eef_quat" in observation:
            sample["eef_quaternion_xyzw"] = np.asarray(observation["robot0_eef_quat"]).copy()
        validate_sample(sample, require_depth=False)
        self.camera(sample, 'wrist')
        return sample

    def camera(self, sample, view="agentview"):
        if view not in {"agentview", "wrist"}:
            raise ValueError("Unknown calibrated camera view")
        intrinsic_key = "intrinsic" if view == "agentview" else "wrist_intrinsic"
        transform_key = "camera_to_world" if view == "agentview" else "wrist_camera_to_world"
        intrinsic = np.array(sample[intrinsic_key], dtype=np.float64, copy=True)
        intrinsic[:2, 2] += self.contract["principal_point_offset_pixels"]
        return Camera(intrinsic, sample[transform_key], (256, 256))


def rigid_surface_points_from_sim(sim, ee_world_m, ee_rotation_world,
                                  body_name="gripper0_right_gripper", max_points=256):
    """Read deterministic triangle centroids from a rigid robot body's GT mesh.

    These points are independent of depth. Finger geoms are excluded because
    their motion is not rigid relative to the EE. No model or state is modified.
    MuJoCo compiled vertices already include mesh scale/pose normalization.
    """
    if max_points < 3:
        raise ValueError("At least three candidate surface points required")
    body = sim.model.body_name2id(body_name)
    collected = []
    for geom in range(sim.model.ngeom):
        mesh = int(sim.model.geom_dataid[geom])
        if int(sim.model.geom_bodyid[geom]) != body or int(sim.model.geom_type[geom]) != 7 or mesh < 0:
            continue
        v0, nv = int(sim.model.mesh_vertadr[mesh]), int(sim.model.mesh_vertnum[mesh])
        f0, nf = int(sim.model.mesh_faceadr[mesh]), int(sim.model.mesh_facenum[mesh])
        vertices = np.asarray(sim.model.mesh_vert[v0:v0+nv], dtype=np.float64)
        faces = np.asarray(sim.model.mesh_face[f0:f0+nf], dtype=int)
        centres = vertices[faces].mean(axis=1)
        rotation = np.asarray(sim.data.geom_xmat[geom]).reshape(3, 3)
        world = centres @ rotation.T + sim.data.geom_xpos[geom]
        collected.append((world - np.asarray(ee_world_m)) @ np.asarray(ee_rotation_world))
    if not collected:
        raise ValueError(f"No rigid triangle mesh found on {body_name}")
    points = np.unique(np.round(np.concatenate(collected), 10), axis=0)
    indices = np.linspace(0, len(points)-1, min(max_points, len(points)), dtype=int)
    return points[indices]
