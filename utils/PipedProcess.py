from .BufferConnection import BufferConnection
import torch.multiprocessing as mp
import asyncio

class Process:
    def __init__(self):
        self.process = None
        self.started = False

    def _createProcess(self):
        self.process = mp.Process(target=self.run, args=())
        self.process.start()

    def _createLoop(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.asyncRun())
        loop.close()

    def start(self):
        if self.started:
            raise Exception("Process is started")

        self.started = True
        self._createProcess()
        return self

    def run(self):
        self._createLoop()

    async def asyncRun(self):
        pass


class PipedProcess(Process):
    def __init__(self):
        super().__init__()

    def _createProcess(self):
        connections = mp.Pipe(True)
        self.conn = BufferConnection(connections[0])
        child_conn = BufferConnection(connections[1])
        self.process = mp.Process(target=self.run, args=(child_conn,))
        self.process.start()

    def _createLoop(self, conn):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.asyncRun(conn))
        loop.close()

    def poll(self):
        return self.conn.poll()

    def recv(self):
        return self.conn.recv()

    def send(self, obj):
        return self.conn.send(obj)

    def run(self, conn):
        self._createLoop(conn)

    async def asyncRun(self, conn):
        pass
