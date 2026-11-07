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
