# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Quadrotor reference/PID interface for article VI-A/B.

The adapter adds mass-times-acceleration feedforward and optional force limits
to the pinned simulator PID. The flight driver chooses feedforward/limits
per experiment and records requested and applied forces separately."""
import math

import numpy as np
import pybullet as bullet
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class AccelerationPIDControl(DSLPIDControl):
    def __init__(self, drone_model, *, feedforward=True, max_tilt_degrees=None):
        super().__init__(drone_model)
        self.mass = self._getURDFParameter("m")
        if max_tilt_degrees is not None and not 0 < max_tilt_degrees < 85:
            raise ValueError("Tilt limit must be between 0 and 85 degrees")
        self.feedforward = feedforward
        self.max_tilt_degrees = max_tilt_degrees
        self.target_acceleration = np.zeros(3)
        self.diagnostics = {}

    def computeControl(self, *args, target_acceleration=None, **kwargs):
        value = np.zeros(3) if target_acceleration is None else np.asarray(target_acceleration, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Expected a finite acceleration vector")
        self.target_acceleration = value
        result = super().computeControl(*args, **kwargs)
        rpm = result[0]
        minimum = self.PWM2RPM_SCALE*self.MIN_PWM+self.PWM2RPM_CONST
        maximum = self.PWM2RPM_SCALE*self.MAX_PWM+self.PWM2RPM_CONST
        self.diagnostics.update(motor_saturated=np.isclose(rpm, minimum, atol=1e-6)|np.isclose(rpm, maximum, atol=1e-6))
        return result

    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                               target_pos, target_rpy, target_vel):
        rotation = np.array(bullet.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        position_error = target_pos-cur_pos
        velocity_error = target_vel-cur_vel
        self.integral_pos_e += position_error*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -.15, .15)
        force = (self.P_COEFF_FOR*position_error+self.I_COEFF_FOR*self.integral_pos_e
                 +self.D_COEFF_FOR*velocity_error+np.array([0., 0., self.GRAVITY]))
        if self.feedforward:
            force += self.mass*self.target_acceleration
        requested = force.copy()
        if self.max_tilt_degrees is not None:
            force[2] = max(force[2], .2*self.GRAVITY)
            horizontal = np.linalg.norm(force[:2])
            maximum_xy = force[2]*np.tan(np.radians(self.max_tilt_degrees))
            if horizontal > maximum_xy:
                force[:2] *= maximum_xy/horizontal
            maximum_rpm = self.PWM2RPM_SCALE*self.MAX_PWM+self.PWM2RPM_CONST
            maximum_thrust = .8*4*self.KF*maximum_rpm**2
            force *= min(1., maximum_thrust/np.linalg.norm(force))
        scalar = max(0., np.dot(force, rotation[:, 2]))
        thrust = (math.sqrt(scalar/(4*self.KF))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE
        z_axis = force/np.linalg.norm(force)
        x_heading = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0.])
        y_axis = np.cross(z_axis, x_heading)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        target_rotation = np.vstack([x_axis, y_axis, z_axis]).T
        euler = Rotation.from_matrix(target_rotation).as_euler("XYZ", degrees=False)
        self.diagnostics = dict(requested_force_n=requested, commanded_force_n=force.copy(),
                                target_euler_rad=euler, force_limited=bool(np.linalg.norm(force-requested) > 1e-10),
                                position_error_m=position_error, velocity_error_mps=velocity_error,
                                target_acceleration_mps2=self.target_acceleration.copy())
        return thrust, euler, position_error
