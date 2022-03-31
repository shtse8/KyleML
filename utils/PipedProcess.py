import asyncio
import torch.multiprocessing as mp

from .BufferConnection import BufferConnection


class Process:
    def __init__(self):
        self.process = None
        self.started = False

    def _create_process(self):
        self.process = mp.Process(target=self.run, args=())
        self.process.start()

    def _create_loop(self):
        asyncio.run(self.async_run())

    def start(self):
        if self.started:
            raise Exception("Process is started")

        self.started = True
        self._create_process()
        return self

    def run(self):
        self._create_loop()

    async def async_run(self):
        pass


class PipedProcess(Process):
    def __init__(self):
        super().__init__()

    def _create_process(self):
        connections = mp.Pipe(True)
        self.conn = BufferConnection(connections[0])
        child_conn = BufferConnection(connections[1])
        self.process = mp.Process(target=self.run, args=(child_conn,))
        self.process.start()

    def _create_loop(self, conn):
        asyncio.run(self.async_run(conn))

    def poll(self):
        return self.conn.poll()

    def recv(self):
        return self.conn.recv()

    def send(self, obj):
        return self.conn.send(obj)

    def run(self, conn):
        self._create_loop(conn)

    async def async_run(self, conn):
        pass
