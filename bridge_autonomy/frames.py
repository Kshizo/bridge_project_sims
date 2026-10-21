"""Frame conversions between ArduPilot (NED / FRD) and the map frame (ENU / FLU).

ArduPilot's local origin is the EKF origin (SITL home = the world's <spherical_coordinates>), so
NED (north, east, down) maps to the world ENU frame as x = east, y = north, z = up.
"""
import math
import numpy as np

_M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)    # NED <-> ENU (its own inverse)
_F = np.diag([1.0, -1.0, -1.0])                              # FRD <-> FLU


def ned_to_enu(n, e, d):
    return np.array([e, n, -d])


def enu_to_ned(p):
    return p[1], p[0], -p[2]


def att_to_R(roll, pitch, yaw):
    """ArduPilot ATTITUDE (NED/FRD euler) -> rotation body FLU -> world ENU."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    R_nf = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    return _M @ R_nf @ _F
