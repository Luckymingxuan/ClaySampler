"""Check an extracted Windows release with the libraries shipped inside it."""

import argparse
import ctypes
import hashlib
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import wave
import zipfile

from check_tail import RATE, midi, read_wav


def check_nvrtc(directory):
    with os.add_dll_directory(directory):
        builtins = ctypes.WinDLL(str(directory / "nvrtc-builtins64_124.dll"))
        rtc = ctypes.WinDLL(str(directory / "nvrtc64_120_0.dll"))
        major, minor = ctypes.c_int(), ctypes.c_int()
        assert rtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0
        assert (major.value, minor.value) == (12, 4)

        program = ctypes.c_void_p()
        code = b'extern "C" __global__ void smoke(int* output) { *output = 42; }'
        assert rtc.nvrtcCreateProgram(ctypes.byref(program), code, b"smoke.cu", 0, None, None) == 0
        try:
            flags = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_75")
            if rtc.nvrtcCompileProgram(program, len(flags), flags):
                size = ctypes.c_size_t()
                rtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                rtc.nvrtcGetProgramLog(program, log)
                raise AssertionError(log.value.decode(errors="replace"))
        finally:
            rtc.nvrtcDestroyProgram(ctypes.byref(program))


def check(archive, root, gpu):
    checksum_file = archive.with_name("SHA256SUMS.txt")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checksum_file.read_text(encoding="ascii").split() == [digest, archive.name]
    directory = root / "extracted"
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(directory)

    executable = directory / "claysampler.exe"
    with executable.open("rb") as binary:
        binary.seek(0x3C)
        pe_offset, = struct.unpack("<I", binary.read(4))
        binary.seek(pe_offset)
        assert binary.read(6) == b"PE\x00\x00\x64\x86", "expected a Windows x64 executable"

    fixture = root / "fixtures"
    fixture.mkdir()
    with wave.open(str(fixture / "sample.wav"), "wb") as sample:
        sample.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        sample.writeframes(struct.pack("<h", 4000) * RATE)
    preset = fixture / "smoke.sfz"
    preset.write_text("<region> sample=sample.wav key=60 loop_mode=one_shot amp_veltrack=0\n", encoding="ascii")
    source = fixture / "smoke.mid"
    midi(source, bytes.fromhex("00 90 3c 64 83 60 80 3c 00 00 ff 2f 00"))

    env = {key: value for key, value in os.environ.items() if key not in {"CLAY_NVRTC", "CLAY_SNDFILE"}}

    def run(*args):
        result = subprocess.run([executable, *args], cwd=fixture, env=env, check=False,
                                capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    assert "Usage:" in run("--help")
    assert "SFZ\tOK\t1" in run("--check-sfz", preset)
    # Windows keeps loaded DLLs mapped until the process exits.
    probe = multiprocessing.get_context("spawn").Process(target=check_nvrtc, args=(directory,))
    probe.start()
    probe.join()
    assert probe.exitcode == 0, "bundled NVRTC check failed"
    if gpu:
        output = fixture / "rendered"
        assert "OK\tsmoke.wav" in run("--sfz", preset, "--output", output, "--quality-gate", "clayengine", source)
        channels, frames, samples = read_wav(output / "smoke.wav")
        assert channels == 1 and frames > RATE // 2 and any(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--gpu", action="store_true", help="also run a CUDA render using only the packaged DLLs")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="claysampler-package-check-") as temp:
        check(args.archive.resolve(), Path(temp), args.gpu)
    print("Windows package checks passed")
