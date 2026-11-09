# Windows releases

The Windows workflow runs on branch pushes and pull requests to `main`. It builds ClaySampler with pinned Zig, packages the executable and runtime DLLs, and checks the extracted ZIP. Each run makes the ZIP and `SHA256SUMS.txt` available in the `claysampler-win64` Actions artifact.

The ZIP includes NVIDIA NVRTC 12.4.127 with its matching builtins and libsndfile 1.2.2. The packager checks the SHA-256 of each downloaded dependency and includes third-party notices and source materials. An installed NVIDIA driver is required for rendering.

The CI check launches the packaged executable from another working directory, checks that libsndfile can open an SFZ sample, and compiles a small CUDA kernel with the packaged NVRTC and builtins. On a Windows CUDA machine, use `--gpu` to render a short clip through the extracted executable.

```powershell
python -B tests/check_windows_package.py build/release/claysampler-dev-win64.zip --gpu
```

To package an existing Windows x64 build locally, use Python 3.11 or newer and provide the path to the compiled executable.

```powershell
python -B scripts/package_windows.py --exe build/cmake/claysampler.exe --version dev --output build/release
```

## Publish

After the release workflow has merged and passed on `main`, sync the branch and tag the commit that should be released.

```powershell
git switch main
git pull --ff-only
git tag v0.1.0
git push origin refs/tags/v0.1.0
```

A pushed `vMAJOR.MINOR.PATCH` tag starts a fresh build. After the package passes verification, the publishing job checks that the tagged commit is on `main` and verifies the ZIP's SHA-256 again. For `v0.1.0`, the Release assets are `claysampler-v0.1.0-win64.zip` and `SHA256SUMS.txt`. The Release body combines reusable installation text from `.github/release-notes.md` with GitHub's generated change list.

For a pinned installer, record the Release asset URL and the matching value from `SHA256SUMS.txt`. Verify the downloaded ZIP against the pinned hash before extracting.
