"""Spatial ops from real positions (scene.json, written by a scene scan): delays by distance and fulldome waves.

Units are Unreal's: centimetres, rotations [pitch, yaw, roll] in degrees. Channel i = actors[i] (after "@group"
expansion). Every delay is >= 0 (the first actor reached fires on time), so these ops are causal.

Dome convention (UE FRotator, rotation matrix applied roll, then pitch, then yaw): the camera's forward axis (+X after
rotation) is the dome centre. zenith = angle between an actor and that axis (0 at the centre, 90 on the rim of a
180-degree dome). azimuth = angle around the axis, 0 towards the camera's up (+Z) and 90 towards its right (+Y): the
clockwise direction for a viewer looking up into the dome with the camera's up at the top.
"""
import numpy as np

from .ops import OpError, P, REQ, op, same_kind
from .signal import delay as delay_sig

ZENITH_DIRECTIONS = ["out", "in"]
AZIMUTH_DIRECTIONS = ["cw", "ccw"]
SHAPES = ["point", "axis", "plane"]


def actor_positions(ctx, node, key="actors"):
    """[C, 3] locations (cm) for the node's actors from the rig's scene"""
    actors = node.get(key) or []
    sc = (ctx.scene or {}).get("actors")
    if not sc:
        raise OpError(node.get("id"), "needs a scene with actor positions (rig \"scene\": \"scene.json\")")
    missing = [a for a in actors if a not in sc or not sc[a].get("location")]
    if missing:
        raise OpError(node.get("id"), f"actors not in the scene (or without a location): {missing[:6]}")
    return np.array([sc[a]["location"] for a in actors], np.float64)


def knn_hops(pos, k, max_hops):
    """[C, C] graph distance in hops (-1 = farther than max_hops) in the symmetric graph that joins every actor to its
    k nearest (ties broken by channel order, so the graph is deterministic)"""
    C = len(pos)
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    A = np.zeros((C, C), bool)
    if C > 1:
        near = np.argsort(D, axis=1, kind="stable")[:, :max(1, min(k, C - 1))]
        A[np.repeat(np.arange(C), near.shape[1]), near.ravel()] = True
        A |= A.T
    hop = np.full((C, C), -1, np.int64)
    np.fill_diagonal(hop, 0)
    reached = np.eye(C, dtype=bool)
    front = reached.copy()
    for d in range(1, int(max_hops) + 1):
        nxt = (front.astype(np.int64) @ A.astype(np.int64) > 0) & ~reached
        if not nxt.any():
            break
        hop[nxt] = d
        reached |= nxt
        front = nxt
    return hop


def rot_axes(rotation):
    """UE FRotator [pitch, yaw, roll] (deg) -> world-space (forward, right, up) unit vectors"""
    p, y, r = np.radians(np.asarray(rotation, np.float64)[:3])
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return fwd, right, up


def dome_angles(camera, pos):
    """(zenith deg, azimuth deg 0..360) of each position seen from a dome camera {location, rotation}.

    A domemaster camera (e.g. OWL 360 in Domemaster mode) looks along the camera actor's UP axis: with rotation (0,0,0)
    the dome's zenith is world +Z and objects on the horizon land on the circle's rim. Pitching the actor tilts the dome
    (a tilted theatre dome). Verified 2026-11-07 against an OWL 360 render: 8 lights spaced 45 deg apart at 6 deg above
    the horizon rendered evenly round the rim. Azimuth is measured from the camera's forward axis toward its right axis.
    """
    loc = np.asarray(camera.get("location") or [0, 0, 0], np.float64)
    fwd, right, up = rot_axes(camera.get("rotation") or [0, 0, 0])
    rel = np.asarray(pos, np.float64) - loc
    x, yv, z = rel @ fwd, rel @ right, rel @ up
    zen = np.degrees(np.arctan2(np.hypot(x, yv), z))
    az = np.mod(np.degrees(np.arctan2(yv, x)), 360.0)
    return zen, az


def _speed_frames(node, ctx, units, per_s_key, per_beat_key):
    """[C] amounts (cm or degrees) -> [C, N] delay frames for a speed in units/s or beats per unit"""
    def f(amount):
        a = np.asarray(amount, np.float64)[:, None]
        if node.get(per_s_key) is not None:
            sp = float(node[per_s_key])
            if sp <= 0:
                raise OpError(node.get("id"), f"{per_s_key} must be > 0")
            return np.broadcast_to(a / sp * ctx.fps, (len(a), ctx.frames))
        if node.get(per_beat_key) is not None:
            if float(node[per_beat_key]) <= 0:
                raise OpError(node.get("id"), f"{per_beat_key} must be > 0")
            return a * units(float(node[per_beat_key])) * ctx.beats_frames(1.0)[None, :]
        raise OpError(node.get("id"), f"needs {per_s_key} or {per_beat_key}")
    return f


def _origin(node, ctx):
    o = node.get("origin")
    if isinstance(o, str):
        sc = (ctx.scene or {}).get("actors") or {}
        if o not in sc or not sc[o].get("location"):
            raise OpError(node.get("id"), f"origin actor {o!r} not in the scene")
        return np.asarray(sc[o]["location"], np.float64)
    if isinstance(o, (list, tuple)) and len(o) == 3:
        return np.asarray(o, np.float64)
    raise OpError(node.get("id"), "origin must be an actor name or [x, y, z] (cm)")


@op("spatial_delay", {"actors": REQ, "origin": REQ, "shape": "point", "dir": None, "speed_cm_s": None, "beats_per_m": None,
                      "direction": "out", "relative": True},
    kind=same_kind, chans=lambda n, c: len(n.get("actors") or []), one_of=(("speed_cm_s", "beats_per_m"),), color=True)
def op_spatial_delay(node, ins, ctx):
    """a wave through real positions: each actor is delayed by its distance from the origin divided by the speed.
    shape: "point" (distance from origin), "axis" (distance from the line through origin along dir), "plane" (signed
    distance along dir, i.e. a sweep across the plane through origin). direction "out" = near first, "in" = far
    first. relative (default): the first actor reached fires with no delay. speed_cm_s, or beats_per_m (tempo-locked)."""
    s = ins[0]
    pos = actor_positions(ctx, node)
    C = len(pos)
    if s.channels not in (1, C):
        raise OpError(node.get("id"), f"input has {s.channels} channels, expected 1 or {C} (one per actor)")
    rel = pos - _origin(node, ctx)
    shape = P(node, "shape")
    if shape == "point":
        d = np.linalg.norm(rel, axis=1)
    else:
        u = np.asarray(node.get("dir") or [0, 0, 1], np.float64)
        if np.linalg.norm(u) == 0:
            raise OpError(node.get("id"), "dir must be a non-zero vector")
        u = u / np.linalg.norm(u)
        along = rel @ u
        d = np.linalg.norm(rel - along[:, None] * u[None, :], axis=1) if shape == "axis" else along
    if shape not in SHAPES:
        raise OpError(node.get("id"), f"shape {shape!r} not in {SHAPES}")
    direction = P(node, "direction")
    if direction == "in":
        d = d.max() - d
    elif direction == "out":
        d = d - d.min() if (P(node, "relative") or shape == "plane") else np.abs(d)
    else:
        raise OpError(node.get("id"), f"direction {direction!r} not in {ZENITH_DIRECTIONS}")
    D = _speed_frames(node, ctx, lambda b: b / 100.0, "speed_cm_s", "beats_per_m")(d)
    return delay_sig(s.broadcast(C), D)


def dome_camera(node, ctx):
    cam = node.get("camera") or (ctx.scene or {}).get("camera")
    if not cam:
        raise OpError(node.get("id"), "needs a dome camera: scene.json \"camera\" {location, rotation} or the node's \"camera\"")
    return cam


@op("dome_wave", {"actors": REQ, "mode": "zenith_rings", "deg_per_s": None, "deg_per_beat": None, "direction": None,
                  "start_deg": 0.0, "relative": True, "camera": None},
    kind=same_kind, chans=lambda n, c: len(n.get("actors") or []), one_of=(("deg_per_s", "deg_per_beat"),), color=True)
def op_dome_wave(node, ins, ctx):
    """a wave over a (tilted) fulldome, from each actor's angle as seen by the dome camera.
    zenith_rings: delay by angle from the dome centre (direction "out" = centre first, "in" = rim first).
    azimuth_sweep: delay by angle around the dome from start_deg (direction "cw" or "ccw").
    speed: deg_per_s, or deg_per_beat (tempo-locked; 360 / deg_per_beat beats per turn)."""
    s = ins[0]
    pos = actor_positions(ctx, node)
    C = len(pos)
    if s.channels not in (1, C):
        raise OpError(node.get("id"), f"input has {s.channels} channels, expected 1 or {C} (one per actor)")
    zen, az = dome_angles(dome_camera(node, ctx), pos)
    mode = P(node, "mode")
    if mode == "zenith_rings":
        direction = node.get("direction") or "out"
        if direction not in ZENITH_DIRECTIONS:
            raise OpError(node.get("id"), f"direction {direction!r} not in {ZENITH_DIRECTIONS}")
        a = zen.max() - zen if direction == "in" else (zen - zen.min() if P(node, "relative") else zen)
    elif mode == "azimuth_sweep":
        direction = node.get("direction") or "cw"
        if direction not in AZIMUTH_DIRECTIONS:
            raise OpError(node.get("id"), f"direction {direction!r} not in {AZIMUTH_DIRECTIONS}")
        a0 = float(P(node, "start_deg"))
        a = np.mod(az - a0, 360.0) if direction == "cw" else np.mod(a0 - az, 360.0)
        if P(node, "relative"):
            a = a - a.min()
    else:
        raise OpError(node.get("id"), f"mode {mode!r} not in ['zenith_rings', 'azimuth_sweep']")
    D = _speed_frames(node, ctx, lambda b: 1.0 / b, "deg_per_s", "deg_per_beat")(a)
    return delay_sig(s.broadcast(C), D)


def outside_dome(camera, scene_actors, actors):
    """actors whose zenith angle is past the dome's edge (dome_angle / 2) -> [(name, zenith)]"""
    pos = [scene_actors[a]["location"] for a in actors if a in scene_actors and scene_actors[a].get("location")]
    names = [a for a in actors if a in scene_actors and scene_actors[a].get("location")]
    if not pos:
        return []
    zen, _ = dome_angles(camera, np.array(pos, np.float64))
    lim = float(camera.get("dome_angle", 180)) / 2
    return [(n, round(float(z), 1)) for n, z in zip(names, zen) if z > lim + 1e-6]


