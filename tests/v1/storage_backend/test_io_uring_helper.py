# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import sys
import time
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.io_uring_helper import (
    IoUringContext,
    _future_registry,
)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


class TestIouringHelper:
    """Test cases for io_uring context."""

    def test_uring_context_initialisation(self, async_loop):
        """ IoUringContext correctly creates an io_uring ring """
        ctx = IoUringContext(loop=async_loop, entries=32, cqsize=32)

        # Check that the internal objects were created (accessing private members directly)
        assert hasattr(ctx, "ring")
        assert hasattr(ctx, "_poll_thread")
        """ The background poll thread is started and alive """
        assert ctx._poll_thread.is_alive()

        ctx.close_all()

    def test_future_registry_roundtrip(self, async_loop):
        """ Test the future registery behaviour """
        ctx = IoUringContext(loop=async_loop, entries=8)

        # Create a Future and insert it into the global registry
        fut = async_loop.create_future()
        _future_registry[id(fut)] = fut

        ctx._resolve_future(id(fut), result=123)
        assert fut.done() and fut.result() == 123

        # Error path (negative error code)
        fut_err = async_loop.create_future()
        _future_registry[id(fut_err)] = fut_err
        ctx._resolve_future(id(fut_err), result=-5)
        assert fut_err.done()
        exc = fut_err.exception()
        assert isinstance(exc, OSError)
        assert exc.errno == 5
        assert exc.strerror == os.strerror(5)

        # no exception expected
        ctx._resolve_future(999999, result=0)

        ctx.close_all()

    def test_poll_thread_stops_gracefully(self, async_loop):
        """ Test background poll thread stops gracefully """
        ctx = IoUringContext(loop=async_loop, entries=8)
        assert not ctx._stop.is_set()

        ctx.close_all()
        assert ctx._stop.is_set()
        assert not ctx._poll_thread.is_alive()

    @pytest.mark.asyncio
    async def test_end_to_end_write_read_real_file(self, tmp_path):
        """ Test that a real file can be written and then read back """
        loop = asyncio.get_running_loop()
        ctx = IoUringContext(loop, entries=8)

        file_path = tmp_path / "uring_test.bin"
        data = b"LMCache IoUring integration test!"

        # write
        written = await ctx.write(str(file_path), data, use_odirect=False)
        assert written == len(data)

        # read
        buf = bytearray(len(data))
        read_len = await ctx.read(str(file_path), buf, use_odirect=False)
        assert read_len == len(data)
        assert bytes(buf) == data

        ctx.close_all()
