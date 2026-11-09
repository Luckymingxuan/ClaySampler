"""Build a Windows release ZIP from a compiled ClaySampler executable."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NVRTC = "cuda_nvrtc-windows-x86_64-12.4.127-archive"
SNDFILE = "libsndfile-1.2.2"
CUDA_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64"
SNDFILE_RELEASE = "https://github.com/libsndfile/libsndfile/releases/download/1.2.2"
CODELOAD = "https://codeload.github.com"

# Match the codecs included in the upstream Windows DLL. The associated vcpkg
# ports provide the codec build recipes and patches.
SOURCES = (
    (f"{SNDFILE}.tar.xz", f"{SNDFILE_RELEASE}/{SNDFILE}.tar.xz",
     "3799ca9924d3125038880367bf1468e53a1b7e3686a934f098b7e1d286cdb80e", "libsndfile-1.2.2/COPYING"),
    ("flac-1.4.2.tar.gz", f"{CODELOAD}/xiph/flac/tar.gz/refs/tags/1.4.2",
     "8e8e0406fb9e1d177bb4ba8cfed3ca3935d37144eac8f0219a03e8c1ed5cc18e", "flac-1.4.2/COPYING.Xiph"),
    ("ogg-1.3.5.tar.gz", f"{CODELOAD}/xiph/ogg/tar.gz/refs/tags/v1.3.5",
     "f6f1b04cfa4e98b70ffe775d5e302d9c6b98541f05159af6de2d6617817ed7d6", "ogg-1.3.5/COPYING"),
    ("vorbis-1.3.7.tar.gz", f"{CODELOAD}/xiph/vorbis/tar.gz/refs/tags/v1.3.7",
     "270c76933d0934e42c5ee0a54a36280e2d87af1de3cc3e584806357e237afd13", "vorbis-1.3.7/COPYING"),
    ("opus-1.4.tar.gz", f"{CODELOAD}/xiph/opus/tar.gz/refs/tags/v1.4",
     "659e6b223e42a51b0a898632b9a5f406ccd5c2e00aa526ddd1264789774b94e5", "opus-1.4/COPYING"),
    ("mpg123-1.31.3.tar.bz2", "https://distfiles.macports.org/mpg123/mpg123-1.31.3.tar.bz2",
     "1ca77d3a69a5ff845b7a0536f783fee554e1041139a6b978f6afe14f5814ad1a", "mpg123-1.31.3/COPYING"),
    ("lame-3.100.tar.gz", "https://deb.debian.org/debian/pool/main/l/lame/lame_3.100.orig.tar.gz",
     "ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e", "lame-3.100/COPYING"),
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(destination, url, expected):
    print(f"Fetching {destination.name}", flush=True)
    with urlopen(url, timeout=60) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if sha256(destination) != expected:
        raise ValueError(f"SHA-256 mismatch for {destination.name}")
    return destination


def copy_zip_member(archive, member, destination):
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def add_runtime(stage, work):
    licenses = stage / "third-party" / "licenses"
    licenses.mkdir(parents=True)
    nvrtc = download(work / f"{NVRTC}.zip", f"{CUDA_REDIST}/{NVRTC}.zip",
                     "f140545e06d0d10780c1382a577db2e2c242db7a2d94970f0e6026b2d01aeb1b")
    with zipfile.ZipFile(nvrtc) as archive:
        for dll in ("nvrtc64_120_0.dll", "nvrtc-builtins64_124.dll"):
            copy_zip_member(archive, f"{NVRTC}/bin/{dll}", stage / dll)
        copy_zip_member(archive, f"{NVRTC}/LICENSE", licenses / "NVIDIA-CUDA.txt")

    sndfile = download(work / f"{SNDFILE}-win64.zip", f"{SNDFILE_RELEASE}/{SNDFILE}-win64.zip",
                       "2173935c0c1ed13cf627951d34483f9d405ead2eb473190461c42ba220643a3f")
    with zipfile.ZipFile(sndfile) as archive:
        copy_zip_member(archive, f"{SNDFILE}-win64/bin/sndfile.dll", stage / "libsndfile-1.dll")

    sources = stage / "third-party" / "sources"
    sources.mkdir()
    for name, url, digest, license_member in SOURCES:
        source = download(sources / name, url, digest)
        with tarfile.open(source) as archive:
            (licenses / f"{license_member.split('/')[0]}.txt").write_bytes(
                archive.extractfile(license_member).read()
            )

    vcpkg = download(work / "vcpkg-2023.08.09.tar.gz", f"{CODELOAD}/microsoft/vcpkg/tar.gz/refs/tags/2023.08.09",
                     "241ae285fe27ce3f8478529d9a05e7a49396789fd8618ed62a11c3e4835aaab2")
    ports = {"libogg", "libvorbis", "libflac", "opus", "mpg123", "mp3lame"}
    with tarfile.open(vcpkg) as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and len(parts) >= 4 and parts[1] == "ports" and parts[2] in ports:
                destination = sources / "vcpkg-2023.08.09" / Path(*parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
        (licenses / "vcpkg.txt").write_bytes(archive.extractfile("vcpkg-2023.08.09/LICENSE.txt").read())

    (stage / "third-party" / "README.md").write_text(
        "# Bundled runtime\n\n"
        f"- NVIDIA NVRTC 12.4.127 and matching builtins from the "
        f"[NVIDIA redistributable archive]({CUDA_REDIST}/{NVRTC}.zip).\n"
        f"- libsndfile 1.2.2 from the [upstream Windows build]({SNDFILE_RELEASE}/{SNDFILE}-win64.zip). "
        "ClaySampler loads this DLL dynamically under the name `libsndfile-1.dll`.\n"
        "- The source archives and vcpkg codec recipes are in `sources/`. "
        "Third-party license texts are in `licenses/`.\n",
        encoding="utf-8",
    )


def package(executable, version, output):
    output.mkdir(parents=True, exist_ok=True)
    filename = f"claysampler-{version}-win64.zip"
    with tempfile.TemporaryDirectory(prefix="package-", dir=output) as temporary:
        work = Path(temporary)
        stage = work / "stage"
        stage.mkdir()
        shutil.copyfile(executable, stage / "claysampler.exe")
        for name in ("README.md", "LICENSE"):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / "docs", stage / "docs")
        add_runtime(stage, work)

        archive = work / filename
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(stage).as_posix())
        digest = sha256(archive)
        checksums = work / "SHA256SUMS.txt"
        checksums.write_bytes(f"{digest}  {filename}\n".encode("ascii"))
        archive.replace(output / filename)
        checksums.replace(output / checksums.name)
        print(f"{digest}  {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"dev|v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", args.version):
        parser.error("--version must be dev or vMAJOR.MINOR.PATCH")
    package(args.exe, args.version, args.output)
