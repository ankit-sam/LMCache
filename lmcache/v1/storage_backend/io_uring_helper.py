# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import queue
import threading
import time
from typing import Any, Callable, Sequence

# Third Party
from liburing import *

class UringOp:
    __slots__ = ("op", "fd", "buf", "size", "cb")
    def __init__(self, op: str, fd: int, buf: memoryview, size: int,
                 cb: Callable[[int], None] | None):
        self.op = op
        self.fd = fd
        self.buf = buf
        self.size = size
        self.cb = cb

class IoUringBatchEngine:
    def __init__(self, entries: int, max_batch: int):
        self.ring = io_uring()
        io_uring_queue_init(entries, self.ring, 0)

        self.q: queue.Queue[UringOp | None] = queue.Queue()
        self._stop = False
        self.max_batch = max_batch
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, op: UringOp) -> None:
        self.q.put(op)

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        io_uring_queue_exit(self.ring)

    def _drain_queue(self) -> list[UringOp]:
        ops = []
        for _ in range(self.max_batch):
            try:
                ops.append(self.q.get_nowait())
            except queue.Empty:
                break
        return ops

    def _run(self) -> None:
        while True:
            first = self.q.get()
            if first is None:
                break

            pending = [first] + self._drain_queue()

            for op in pending:
                sqe = io_uring_get_sqe(self.ring)
                if op.op == "write":
                    io_uring_prep_write(sqe, op.fd, op.buf, op.size, 0)
                else:
                    io_uring_prep_read(sqe, op.fd, op.buf, op.size, 0)
                io_uring_sqe_set_data(sqe, op)

            io_uring_submit(self.ring)

            for _ in range(len(pending)):
                cqe = io_uring_cqe()
                io_uring_wait_cqe(self.ring, cqe)
                op: UringOp = io_uring_cqe_get_data(cqe)
                res = cqe.res
                io_uring_cqe_seen(self.ring, cqe)

                if op.cb:
                    os.close(op.fd)
                    op.cb(res)
