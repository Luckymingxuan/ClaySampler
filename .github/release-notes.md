## Windows x64 download

Download the Windows ZIP and `SHA256SUMS.txt` from Assets. Compare the ZIP's SHA-256 with the listed value before extracting.

```powershell
Get-FileHash -Algorithm SHA256 .\claysampler-*-win64.zip
```

Extract the ZIP and keep its DLLs beside `claysampler.exe`. The archive includes NVIDIA NVRTC with its matching builtins and the libsndfile audio decoder. Rendering requires a CUDA-capable NVIDIA GPU and a compatible NVIDIA driver.

```powershell
.\claysampler.exe --sfz instrument.sfz --output renders take1.mid
```

See the included README for batch manifests, tail policies and optional GPU audio checks. Third-party licenses and sources are in the `third-party` directory.
