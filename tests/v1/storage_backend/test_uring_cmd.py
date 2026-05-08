# SPDX-License-Identifier: Apache-2.0

"""Tests for io_uring command (passthrough) support in Rust raw block backend."""

# Standard
from unittest.mock import MagicMock, patch
import asyncio
import os

# Third Party
import pytest

# First Party
from lmcache.logging import init_logger
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
