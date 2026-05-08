# SPDX-License-Identifier: Apache-2.0

"""Tests for io_uring command (passthrough) support in Rust raw block backend."""

# Standard
from unittest.mock import MagicMock, patch
import asyncio
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
import lmcache.v1.storage_backend.plugins.rust_raw_block_backend as raw_block_backend

logger = init_logger(__name__)


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


class MockConfig:
    """Mock configuration for testing."""

    def __init__(
        self,
        device_path: str,
        use_uring_cmd: bool = False,
        meta_total_bytes=4 * 1024 * 1024,
    ):
        self.extra_config = {
            "rust_raw_block.device_path": device_path,
            "rust_raw_block.use_odirect": False,
            "rust_raw_block.use_uring": True,
            "rust_raw_block.use_uring_cmd": use_uring_cmd,
            "rust_raw_block.capacity_bytes": 1024 * 1024 * 1024,  # 1GB
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": meta_total_bytes,
        }


class MockMetadata:
    """Mock metadata for testing."""

    def __init__(self, worker_id: int = 0, world_size: int = 1):
        self.worker_id = worker_id
        self.world_size = world_size


class MockLocalCPUBackend:
    """Mock local CPU backend for testing."""

    def __init__(self):
        pass

    def get_memory_allocator(self):
        return None

    def get_full_chunk_size_bytes(self) -> int:
        """return a default chunk size only for testing."""
        return 256 * 1024


class _FakeFdpRawDevice:
    def __init__(
        self,
        fdp_status: list[tuple[int, int]] | None = None,
        fetch_error: Exception | None = None,
    ):
        self.fdp_status = fdp_status or []
        self.fetch_error = fetch_error
        self.write_uring = MagicMock()
        self.batched_write = MagicMock(return_value=1)
        self.wait_iouring = MagicMock()

    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.fdp_status

    def close(self) -> None:
        return None


def _build_rust_raw_block_fdp_config(
    dev_path: str,
    extra: dict[str, object] | None = None,
) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="test_rust_raw_block_backend_fdp",
    )
    config.storage_plugins = []
    config.extra_config = {
        "rust_raw_block.device_path": dev_path,
        "rust_raw_block.block_align": 4096,
        "rust_raw_block.header_bytes": 4096,
        "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
        "rust_raw_block.meta_enable_periodic": False,
        "rust_raw_block.max_data_transfer_size": 0,
        "rust_raw_block.use_uring": True,
        "rust_raw_block.use_uring_cmd": True,
        "rust_raw_block.use_fdp": True,
        **(extra or {}),
    }
    return config


def _build_rust_raw_block_metadata(
    worker_id: int = 0,
    world_size: int = 1,
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _build_rust_raw_block_local_cpu_backend() -> MagicMock:
    local_cpu_backend = MagicMock()
    local_cpu_backend.get_full_chunk_size_bytes.return_value = 4096
    return local_cpu_backend


def _build_fdp_backend(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    raw_device: _FakeFdpRawDevice,
) -> RustRawBlockBackend:
    loop = asyncio.new_event_loop()
    try:
        with (
            patch.object(RustRawBlockBackend, "_rawdev", return_value=raw_device),
            patch.object(RustRawBlockBackend, "_ensure_capacity_and_layout"),
            patch.object(RustRawBlockBackend, "_register_paged_buffers"),
            patch.object(RustRawBlockBackend, "_load_checkpoint_from_device"),
        ):
            backend = RustRawBlockBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=_build_rust_raw_block_local_cpu_backend(),
                loop=loop,
                dst_device="cpu",
            )
            backend._raw = raw_device  # type: ignore[assignment]
            return backend
    finally:
        loop.close()


def _build_fdp_backend_with_loop(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    raw_device: _FakeFdpRawDevice,
    loop: asyncio.AbstractEventLoop,
) -> RustRawBlockBackend:
    with (
        patch.object(RustRawBlockBackend, "_rawdev", return_value=raw_device),
        patch.object(RustRawBlockBackend, "_ensure_capacity_and_layout"),
        patch.object(RustRawBlockBackend, "_register_paged_buffers"),
        patch.object(RustRawBlockBackend, "_load_checkpoint_from_device"),
    ):
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=_build_rust_raw_block_local_cpu_backend(),
            loop=loop,
            dst_device="cpu",
        )
        backend._raw = raw_device  # type: ignore[assignment]
        return backend


def _build_transfer_limit_backend(
    dev_path: str,
    max_data_transfer_size: int | None = None,
) -> RustRawBlockBackend:
    config = MockConfig(device_path=dev_path, use_uring_cmd=False)
    if max_data_transfer_size is not None:
        config.extra_config["rust_raw_block.max_data_transfer_size"] = (
            max_data_transfer_size
        )

    metadata = MockMetadata()
    loop = asyncio.new_event_loop()
    try:
        with (
            patch.object(RustRawBlockBackend, "_rawdev", return_value=MagicMock()),
            patch.object(RustRawBlockBackend, "_ensure_capacity_and_layout"),
            patch.object(RustRawBlockBackend, "_register_paged_buffers"),
            patch.object(RustRawBlockBackend, "_load_checkpoint_from_device"),
        ):
            backend = RustRawBlockBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=MockLocalCPUBackend(),
                loop=loop,
                dst_device="cpu",
            )
            return backend
    finally:
        loop.close()


def test_uring_cmd_requires_character_device(loop_in_thread):
    """Test that io_uring_cmd requires a character device, not a block device."""
    # This test requires a block device device /dev/nvme0n1
    # Skip if this doesn't exist
    device_path = os.environ.get("LMCACHE_TEST_BLOCK_DEVICE", "/dev/nvme0n1")

    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    config = MockConfig(device_path=device_path, use_uring_cmd=True)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()

    # This should raise an error because the device is not a character device
    with pytest.raises(
        ValueError, match="io_uring_cmd requires a NVMe namespace character device"
    ):
        RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )


def test_uring_cmd_get_nvme_info(loop_in_thread):
    """Test getting NVMe namespace ID and LBA size from character device."""
    # This test requires a block device device /dev/nvme0n1
    # Skip if this doesn't exist
    device_path = os.environ.get("LMCACHE_TEST_BLOCK_DEVICE", "/dev/ng0n1")

    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    config = MockConfig(device_path=device_path, use_uring_cmd=True)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()

    try:
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )

        # Get the raw device
        raw_device = backend._rawdev()

        # Test getting namespace ID
        nsid = raw_device.nvme_nsid()
        assert nsid > 0, f"Expected positive nsid, got {nsid}"
        logger.info(f"NVMe namespace ID: {nsid}")

        # Test getting LBA size
        lba_size = raw_device.nvme_lba_size()
        assert lba_size > 0, f"Expected positive lba_size, got {lba_size}"
        logger.info(f"NVMe LBA size: {lba_size} bytes")

    except Exception as e:
        pytest.fail(f"Failed to get NVMe info: {e}")


def test_uring_cmd_disabled(loop_in_thread):
    """Test that NVMe methods are not available when use_uring_cmd is disabled."""
    config = MockConfig(device_path="/dev/null", use_uring_cmd=False)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()
    raw_device = MagicMock()
    raw_device.nvme_nsid.side_effect = RuntimeError("use_uring_cmd not enabled")
    raw_device.nvme_lba_size.side_effect = RuntimeError("use_uring_cmd not enabled")

    with (
        patch.object(RustRawBlockBackend, "_rawdev", return_value=raw_device),
        patch.object(RustRawBlockBackend, "_ensure_capacity_and_layout"),
        patch.object(RustRawBlockBackend, "_register_paged_buffers"),
        patch.object(RustRawBlockBackend, "_load_checkpoint_from_device"),
    ):
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )
        backend._raw = raw_device

    # These should raise errors when use_uring_cmd is disabled
    with pytest.raises(RuntimeError, match="use_uring_cmd not enabled"):
        raw_device.nvme_nsid()

    with pytest.raises(RuntimeError, match="use_uring_cmd not enabled"):
        raw_device.nvme_lba_size()


def test_uring_cmd_fdp_uses_namespace_placement_for_data():
    with tempfile.TemporaryDirectory() as td:
        config = _build_rust_raw_block_fdp_config(os.path.join(td, "dev.bin"))
        metadata = _build_rust_raw_block_metadata()
        backend = _build_fdp_backend(
            config,
            metadata,
            _FakeFdpRawDevice(fdp_status=[(0, 1)]),
        )

    assert backend._get_data_placement_id() == 0
    assert backend._build_data_placement_ids(3) == [0, 0, 0]
    assert len(backend._data_placement_ids_cache) == 3
    assert backend._build_data_placement_ids(2) == [0, 0]
    assert len(backend._data_placement_ids_cache) == 3


def test_uring_cmd_fdp_metadata_uses_default_ruh():
    with tempfile.TemporaryDirectory() as td:
        config = _build_rust_raw_block_fdp_config(os.path.join(td, "dev.bin"))
        metadata = _build_rust_raw_block_metadata()
        raw_device = _FakeFdpRawDevice(fdp_status=[(0, 1)])
        backend = _build_fdp_backend(config, metadata, raw_device)

    assert backend._write_checkpoint(b"{}", dirty_total_snapshot=1)
    for call in raw_device.write_uring.call_args_list:
        assert len(call.args) == 4


@pytest.mark.parametrize(
    "extra",
    [
        {"rust_raw_block.use_uring": False},
        {"rust_raw_block.use_uring_cmd": False},
    ],
)
def test_uring_cmd_fdp_requires_uring_and_uring_cmd(extra: dict[str, object]):
    with tempfile.TemporaryDirectory() as td:
        config = _build_rust_raw_block_fdp_config(os.path.join(td, "dev.bin"), extra)
        metadata = _build_rust_raw_block_metadata()
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RuntimeError, match="use_uring"):
                RustRawBlockBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=_build_rust_raw_block_local_cpu_backend(),
                    loop=loop,
                    dst_device="cpu",
                )
        finally:
            loop.close()


def test_uring_cmd_fdp_fetch_failure_fails_fast():
    with tempfile.TemporaryDirectory() as td:
        config = _build_rust_raw_block_fdp_config(os.path.join(td, "dev.bin"))
        metadata = _build_rust_raw_block_metadata()
        raw_device = _FakeFdpRawDevice(fetch_error=RuntimeError("fdp unavailable"))

        with pytest.raises(RuntimeError, match="failed to fetch FDP status"):
            _build_fdp_backend(config, metadata, raw_device)


def test_uring_cmd_fdp_empty_placement_ids_fail_fast():
    with tempfile.TemporaryDirectory() as td:
        config = _build_rust_raw_block_fdp_config(os.path.join(td, "dev.bin"))
        metadata = _build_rust_raw_block_metadata()
        raw_device = _FakeFdpRawDevice(fdp_status=[])

        with pytest.raises(RuntimeError, match="does not expose any FDP status"):
            _build_fdp_backend(config, metadata, raw_device)


def test_uring_cmd_fdp_data_write_passes_placement_ids():
    with tempfile.TemporaryDirectory() as td:
        config = _build_rust_raw_block_fdp_config(os.path.join(td, "dev.bin"))
        metadata = _build_rust_raw_block_metadata()
        raw_device = _FakeFdpRawDevice(fdp_status=[(0, 1)])
        loop = asyncio.new_event_loop()
        backend = _build_fdp_backend_with_loop(
            config=config,
            metadata=metadata,
            raw_device=raw_device,
            loop=loop,
        )

        try:
            backend._max_slots = 16
            backend._next_slot = 0
            backend._effective_capacity_bytes = (
                backend.meta_total_bytes + backend.slot_bytes * 16
            )

            key = CacheEngineKey("test_model", 1, 0, 4242, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(9)
            obj.ref_count_up()
            loop.run_until_complete(
                backend._batched_submit_put_task_uring([key], [obj])
            )
            obj.ref_count_down()

            raw_device.batched_write.assert_called_once()
            args = raw_device.batched_write.call_args.args
            assert len(args) == 4
            offsets, _buffers, total_lens, placement_ids = args
            assert len(offsets) == len(total_lens) == len(placement_ids)
            assert placement_ids == [0] * len(placement_ids)
        finally:
            backend.close()
            loop.close()


def test_uring_cmd_fdp_rank_based_selects_first_pid():
    with tempfile.TemporaryDirectory() as td:
        config = _build_rust_raw_block_fdp_config(os.path.join(td, "dev.bin"))
        metadata = _build_rust_raw_block_metadata()
        backend = _build_fdp_backend(
            config,
            metadata,
            _FakeFdpRawDevice(fdp_status=[(0, 1), (2, 3)]),
        )

    assert backend._get_data_placement_id() == 0


def test_uring_cmd_auto_transfer_limit_from_sysfs_ng_device():
    expected_path = "/sys/block/nvme0n1/queue/max_hw_sectors_kb"
    with patch.object(
        raw_block_backend,
        "_read_sysfs_int",
        return_value=1024,
    ) as mock_read:
        backend = _build_transfer_limit_backend("/dev/ng0n1", max_data_transfer_size=-1)

    mock_read.assert_called_once_with(expected_path)
    assert backend.max_data_transfer_size == 1024 * 1024


def test_uring_cmd_auto_transfer_limit_fails_when_sysfs_unavailable():
    expected_path = "/sys/block/nvme0n1/queue/max_hw_sectors_kb"
    with patch.object(
        raw_block_backend, "_read_sysfs_int", return_value=None
    ) as mock_read:
        with pytest.raises(RuntimeError, match="failed to read max_hw_sectors_kb"):
            _build_transfer_limit_backend("/dev/ng0n1", max_data_transfer_size=-1)

    mock_read.assert_called_once_with(expected_path)


def test_uring_cmd_auto_transfer_limit_rejects_unsupported_device_path():
    with patch.object(raw_block_backend, "_read_sysfs_int") as mock_read:
        with pytest.raises(
            RuntimeError, match="unable to derive NVMe sysfs queue path"
        ):
            _build_transfer_limit_backend("/dev/nvme0n1", max_data_transfer_size=-1)

    mock_read.assert_not_called()


def test_uring_cmd_auto_transfer_limit_rejects_file_path():
    with patch.object(raw_block_backend, "_read_sysfs_int") as mock_read:
        with pytest.raises(
            RuntimeError, match="unable to derive NVMe sysfs queue path"
        ):
            _build_transfer_limit_backend("/tmp/dev.bin", max_data_transfer_size=-1)

    mock_read.assert_not_called()


def test_uring_cmd_no_split_when_transfer_limit_is_zero():
    backend = _build_transfer_limit_backend("/dev/ng0n1", max_data_transfer_size=0)
    assert backend.max_data_transfer_size == 0


def test_uring_cmd_rejects_invalid_negative_transfer_limit():
    with pytest.raises(
        ValueError, match="max_data_transfer_size must be -1, 0, or > 0"
    ):
        _build_transfer_limit_backend("/dev/ng0n1", max_data_transfer_size=-2)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
