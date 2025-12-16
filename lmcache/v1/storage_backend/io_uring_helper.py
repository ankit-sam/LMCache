# io_uring_helper.py
import os
import threading
import ctypes
import asyncio
from typing import Callable, Any, List

import time
from lmcache.logging import init_logger

from liburing import (
    io_uring,
    io_uring_cqe,
    io_uring_queue_init,
    io_uring_submit,
    io_uring_peek_cqe,
    io_uring_cqe_seen,
    io_uring_cqe_get_data,
    io_uring_register_ring_fd,
    io_uring_queue_exit,
    io_uring_get_sqe,
    io_uring_prep_write,
    io_uring_prep_read,
    io_uring_sqe_set_data,
)

logger = init_logger(__name__)

class IoUringContext:
    """Wraps a single io_uring instance and bridges completions to asyncio."""

    def __init__(self, loop: asyncio.AbstractEventLoop,
                 entries: int = 256,
                 cqsize: int = 256):
        self.loop = loop
        self.ring = io_uring()

        io_uring_queue_init(entries, self.ring, 0)

        # Start a background thread that will reap completions
        self._stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._completion_worker,
                                            name="IoUringPoll",
                                            daemon=True)
        self._poll_thread.start()

        # Simple FD cache for pre-opened files
        # self._fd_cache = {}

    """
    # TODO(Ankit): File descriptor handling (caching)
    def _open_file(self, path: str, flags: int, mode: int) -> int:
        if path in self._fd_cache:
            return self._fd_cache[path]
        fd = os.open(path, flags, mode)
        self._fd_cache[path] = fd
        return fd
    """

    def close_all(self):
        self._stop.set()
        self._poll_thread.join()
        # for fd in self._fd_cache.values():
        #    os.close(fd)
        io_uring_queue_exit(self.ring)

    # Public async API
    async def write(self, path: str, data: bytes, use_odirect: bool = False) -> int:
        """Submit an async write and await its completion."""
        flags = os.O_WRONLY | os.O_CREAT
        if use_odirect:
            flags |= os.O_DIRECT
        fd = os.open(path, flags, 0o644)

        sqe = io_uring_get_sqe(self.ring)
        io_uring_prep_write(sqe, fd, data, len(data), 0)
        # TODO(Ankit): Register the buffer if you want fixed buffer optimization
        # Does it even make sense? As we will register and unregister buffers every time
        # io_uring_register_buffers(self.ring, data, 1)

        # Use user_data to carry a Future that will be resolved later
        fut = self.loop.create_future()
        _future_registry[id(fut)] = fut

        io_uring_sqe_set_data(sqe, id(fut))

        io_uring_submit(self.ring)

        # The poll thread will set the result on the Future
        result = await fut
        os.close(fd)

        return result

    async def read(self, path: str, data: bytes, use_odirect: bool = False) -> int:
        flags = os.O_RDONLY
        if use_odirect:
            flags |= os.O_DIRECT
        fd = os.open(path, flags)

        sqe = io_uring_get_sqe(self.ring)
        io_uring_prep_read(sqe, fd, data, len(data), 0)

        # TODO(Ankit): Register the buffer if you want fixed buffer optimization
        # Does it even make sense? As we will register and unregister buffers every time
        # io_uring_register_buffers(self.ring, data, 1)

        # Use user_data to carry a Future that will be resolved later
        fut = self.loop.create_future()
        _future_registry[id(fut)] = fut
        io_uring_sqe_set_data(sqe, id(fut))

        io_uring_submit(self.ring)
        result = await fut
        os.close(fd)

        return result

    # Completion worker
    def _completion_worker(self):
        """Runs in a background thread, delivering CQEs to asyncio futures."""
        while not self._stop.is_set():
            cqe = io_uring_cqe()
            ret = io_uring_peek_cqe(self.ring, cqe)

            if ret != 0:
                time.sleep(0)
                continue

            if cqe:
                fut_id = io_uring_cqe_get_data(cqe)

                result = cqe.res

                # Resolve the Future on the main loop thread
                self.loop.call_soon_threadsafe(self._resolve_future, fut_id, result)

                # Mark the CQE as seen so the kernel can reuse the slot.
                io_uring_cqe_seen(self.ring, cqe)

    def _resolve_future(self, fut_id: int, result: int):
        """
        Called on the asyncio thread. `fut_id` is the python `id()` of the Future.
        """
        # Retrieve the actual Future object
        # TODO(Ankit): keep a dict: {id(fut): fut}
        # Here we assume a global registry:
        fut = _future_registry.pop(fut_id, None)
        if fut is None:
            return
        if result < 0:
            fut.set_exception(OSError(-result, os.strerror(-result)))
        else:
            fut.set_result(result)

# Global registry used by the helper (simple but works)
_future_registry = {}
