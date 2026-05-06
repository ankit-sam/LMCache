# Examples vLLM + LMCache w. local backends
LMCache should be able to reduce the generation time of the second and following calls.
## CPU offloading
- `python offload.py -v v0` - CPU offloading implementation for vLLM v0
- `python offload.py -v v1` - CPU offloading implementation for vLLM v1
## Disk offloading
- `python offload.py -v v0 --use-disk` - Disk offloading implementation for vLLM v0
- `python offload.py -v v1 --use-disk` - Disk offloading implementation for vLLM v1

## RUST raw block based Disk offloading

   # WARNING: This will erase the content of target device.
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1` - posix disk offloading
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1 --use_uring` - io_uring disk offloading
- `python rust_backend_offload.py --per_tp_device_paths /dev/ng0n1 /dev/ng0n2 --use_fdp --max-data-transfer-size -1` - FDP offloading

FDP notes:
- `--disk_path` can be used for TP=1 FDP runs.
- `--per_tp_device_paths` sets the TP size and assigns one device path per TP rank.
- Each `--per_tp_device_paths` entry must be unique.
- The current FDP placement policy is `rank-based` and is mainly intended for multi-TP runs.
- `--max-data-transfer-size` defaults to `0` (no splitting); set `-1` for auto-detection from `max_hw_sectors_kb`, or provide a positive byte value to force splitting.
