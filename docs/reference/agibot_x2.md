# Agibot X2

`agibot_x2.yml` configures fixed-base dual-arm planning with the X2 Ultra v1.3.0
model. The robot name is accepted by `KinematicsCfg.from_robot_yaml_file`,
`InverseKinematicsCfg.create`, and `MotionPlannerCfg.create`.

| Property | Configuration |
| --- | --- |
| Base frame | `base_link`, coincident with the pelvis frame |
| Tool frames | `left_wrist_roll_link`, `right_wrist_roll_link` |
| Active joints | 14: seven left-arm joints, then seven right-arm joints |
| Joint order within each arm | Shoulder pitch, roll, yaw; elbow; wrist yaw, pitch, roll |
| Locked joints | 12 leg, three waist, and two head joints, all at zero |
| Position and velocity limits | From the URDF, in radians and radians/second |
| Acceleration / jerk limits | Planning defaults: 3 rad/s² and 30 rad/s³ |
| Collision model | 597 spheres across 32 links, with 2.5 mm radius padding |

Goal poses refer to the wrist-roll frames and use `base_link` coordinates. The
model has fixed hand geometry, without articulated fingers or a calibrated grasp
frame. IK solutions and interpolated plans include all 31 movable URDF joints;
`JointState.reorder(planner.joint_names)` selects the 14 active arm joints.

## Collision geometry

Spheres were generated offline with `fit_spheres_to_mesh` and
`SphereFitType.MORPHIT`, comparing whole-link and separate CAD-component convex
envelopes. Increased protrusion penalties reduce outward bulging. Redundant spheres
were pruned, and individual radii were expanded where source surfaces were uncovered.
The head uses 19 spheres and the torso 27. Planning loads the saved fit without refitting.

Source triangles were subdivided to edges of at most 2 mm. Every resulting vertex
is covered before padding; the 2.5 mm radius buffer covers the remaining surface
distance bound plus a 0.5 mm margin. All 32 links were checked in front/side overlays,
with assembled neutral, reaching, and crossed-arm poses and head, torso, and hand
cross-sections. Envelopes still conservatively occupy cavities and some concavities;
the padding value is not a bound on total outward protrusion or a calibrated execution margin.

The model includes every URDF collision link and the visual-only
`waist_yaw_link` and `waist_pitch_link`. Self-collision exclusions cover mechanical
neighbors and overlapping housings. Arm-to-arm and arm-to-body checks remain enabled.
Joint locks define the body posture assumed by collision checking. A different body
posture requires updating those locks. The configuration does not model balance
or walking. Acceleration and jerk limits are planning defaults, not URDF ratings.

See the [sphere-fitting reference][sphere-fitting] for fitting options and the
[robot-model tutorial](../getting-started/build_robot_model.rst) for editing tool
frames and collision geometry.

[sphere-fitting]: https://nvlabs.github.io/curobo/latest/reference/sphere_fitting.html

## Assets

The original URDF, all referenced meshes, provenance, and upstream Mulan PSL v2
license are packaged under `curobo/content/assets/robot/agibot_x2/`. Configuration
paths are package-relative and do not depend on the original asset checkout.
