"""Compare SFZ loading and region counts with the sfizz reference library."""

import argparse
import collections
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path


class Sfizz:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        pointer = ctypes.c_void_p
        self.create = self.bind("create_synth", pointer)
        self.free = self.bind("free", None, pointer)
        self.load = self.bind("load_file", ctypes.c_bool, pointer, ctypes.c_char_p)
        self.preload = self.bind("set_preload_size", None, pointer, ctypes.c_uint)
        self.voices = self.bind("set_num_voices", None, pointer, ctypes.c_int)
        self.regions = self.bind("get_num_regions", ctypes.c_int, pointer)
        self.unknown = self.bind("get_unknown_opcodes", pointer, pointer)
        self.free_memory = self.bind("free_memory", None, pointer)

    def bind(self, name, result, *args):
        function = getattr(self.library, f"sfizz_{name}")
        function.restype = result
        function.argtypes = args
        return function

    def inspect(self, path):
        synth = self.create()
        if not synth:
            raise RuntimeError("sfizz could not create a synth")
        try:
            self.preload(synth, 1)
            self.voices(synth, 8)
            loaded = self.load(synth, path.as_posix().encode("utf-8"))
            regions = self.regions(synth) if loaded else 0
            ptr = self.unknown(synth)
            try:
                text = ctypes.string_at(ptr).decode("utf-8", "replace") if ptr else ""
            finally:
                if ptr:
                    self.free_memory(ptr)
            return {"regions": regions, "unknown": [part.strip() for part in text.split(",") if part.strip()]}
        finally:
            self.free(synth)


def clay_rows(binary, sndfile, paths):
    command = [binary, "--check-sfz"]
    if sndfile:
        command += ["--sndfile", sndfile]
    for start in range(0, len(paths), 24):
        batch = paths[start:start + 24]
        proc = subprocess.run(command + batch, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=False)
        lines = [line.split("\t", 2) for line in proc.stdout.splitlines() if line.startswith("SFZ\t")]
        if proc.returncode not in (0, 2) or len(lines) != len(batch):
            raise RuntimeError(proc.stdout + proc.stderr)
        for _, status, detail in lines:
            yield {"regions": int(detail)} if status == "OK" else {"regions": 0, "reason": detail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay", type=Path, required=True)
    parser.add_argument("--sfizz-library", type=Path, required=True)
    parser.add_argument("--sndfile", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="**/*.sfz")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    clay = options.clay.resolve(strict=True)
    library = options.sfizz_library.resolve(strict=True)
    root = options.root.resolve(strict=True)
    paths = sorted(p for p in root.glob(options.pattern) if p.is_file())
    if not paths:
        parser.error("no SFZ files matched")
    reference = Sfizz(library)
    rows = []
    counts = collections.defaultdict(collections.Counter)
    for path, current in zip(paths, clay_rows(clay, options.sndfile, paths)):
        sfizz = reference.inspect(path)
        relative = path.relative_to(root)
        counts[relative.parts[0]].update({
            "presets": 1,
            "clay_loaded": current["regions"] > 0,
            "sfizz_loaded": sfizz["regions"] > 0,
            "sfizz_with_unknown_opcodes": bool(sfizz["unknown"]),
            "equal_nonzero_region_counts": sfizz["regions"] == current["regions"] > 0,
        })
        rows.append({"preset": relative.as_posix(), "clay": current, "sfizz": sfizz})
    summary = {
        "clay_sha256": hashlib.sha256(clay.read_bytes()).hexdigest(),
        "sfizz_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "libraries": dict(counts),
        "presets": rows,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
