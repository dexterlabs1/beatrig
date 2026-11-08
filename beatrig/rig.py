"""rig.json loading. All relative paths resolve from the rig file's folder."""
import copy
import hashlib
import json
import os
from typing import Optional


class RigError(Exception):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


class Rig:
    def __init__(self, data: dict, base_dir: str, path: Optional[str] = None):
        self.data = copy.deepcopy(data)
        self.base_dir = os.path.abspath(base_dir)
        self.path = os.path.abspath(path) if path else None

    @classmethod
    def from_dict(cls, data: dict, base_dir: str) -> "Rig":
        """a rig held in memory; relative paths resolve from base_dir"""
        if not isinstance(data, dict):
            raise RigError("rig must be a JSON object")
        return cls(data, base_dir)

    def resolve(self, p):
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base_dir, p))

    @property
    def audio_path(self):
        return self.resolve(self.data.get("audio"))

    @property
    def notes_path(self):
        return self.resolve(self.data.get("notes", "notes.json"))

    @property
    def cache_dir(self):
        """where the analysis cache lives: rig.cache_dir (relative to the rig), else the rig's own folder. Scene rigs cut
        from one show mix point at one folder and share one analysis."""
        return self.resolve(self.data.get("cache_dir")) or self.base_dir

    @property
    def start_frame(self):
        """bake.start_frame: the first sequence frame baked (default 0). Frames before it are not evaluated at all."""
        return int((self.data.get("bake") or {}).get("start_frame") or 0)

    @property
    def fps(self):
        return float(self.data.get("fps", 30))

    @property
    def offset(self):
        return int(self.data.get("audio_offset_frames", 0))

    @property
    def seed(self):
        return int(self.data.get("seed", 1))

    @property
    def nodes(self):
        return self.data.get("nodes", [])

    @property
    def outputs(self):
        return self.data.get("outputs", [])

    @property
    def layers(self):
        return self.data.get("layers", [])

    @property
    def hash(self):
        return sha256(self.data)

    def node(self, nid):
        return next((n for n in self.nodes if n.get("id") == nid), None)

    # -- v0.2: audio window, groups, scene
    @property
    def audio_in_s(self):
        """audio time of frame `audio_offset_frames` (a window into a long file); 0 = the file's start"""
        return float(self.data.get("audio_in_s") or 0.0)

    @property
    def audio_out_s(self):
        v = self.data.get("audio_out_s")
        return None if v is None else float(v)

    @property
    def groups(self):
        g = self.data.get("groups")
        return g if isinstance(g, dict) else {}

    def expand_actors(self, actors):
        """an actor list (or one "@group" string) with every "@group" entry replaced by its members, recursively;
        unknown groups stay as written (check reports them)"""
        def walk(items, seen):
            out = []
            for a in ([items] if isinstance(items, str) else list(items or [])):
                if isinstance(a, str) and a.startswith("@") and a[1:] in self.groups and a[1:] not in seen:
                    out.extend(walk(self.groups[a[1:]], seen | {a[1:]}))
                else:
                    out.append(a)
            return out
        return walk(actors, frozenset())

    def resolved_nodes(self):
        """the nodes with "@group" actor lists expanded (what evaluation and channel inference see)"""
        out = []
        for n in self.nodes:
            if isinstance(n, dict) and n.get("actors") is not None:
                n = dict(n, actors=self.expand_actors(n["actors"]))
            out.append(n)
        return out

    def resolved_target(self, target):
        t = dict(target or {})
        if t.get("actors") is not None:
            t["actors"] = self.expand_actors(t["actors"])
        return t

    @property
    def scene_path(self):
        return self.resolve(self.data.get("scene"))

    def load_scene(self):
        """the scene.json dict, or None when the rig has no "scene" (raises RigError if it is set but unreadable)"""
        p = self.scene_path
        if not p:
            return None
        try:
            st = os.stat(p)
            key = (p, st.st_size, st.st_mtime_ns)
            if key not in _SCENES:
                with open(p, encoding="utf-8") as fh:
                    _SCENES[key] = json.load(fh)
            return _SCENES[key]
        except (OSError, json.JSONDecodeError) as e:
            raise RigError(f"scene {p}: {e}") from None


_SCENES = {}


def load_rig(path: str) -> Rig:
    """read rig.json; raises RigError on invalid JSON"""
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as e:
            raise RigError(f"{path}: {e}") from None
    r = Rig.from_dict(data, os.path.dirname(os.path.abspath(path)))
    r.path = os.path.abspath(path)
    return r
