from .BufferConnection import BufferConnection
import torch.multiprocessing as mp


class PipedProcess:
    def __init__(self):
        self.process = None
        self.conn = None
        self.started = False

    def start(self):
        if self.started:
            raise Exception("Process is started")

        self.started = True
        connections = mp.Pipe(True)
        self.conn = BufferConnection(connections[0])
        child_conn = BufferConnection(connections[1])
        self.process = mp.Process(target=self.run, args=(child_conn,))
        self.process.start()
        return self

    def poll(self):
        return self.conn.poll()

    def recv(self):
        return self.conn.recv()

    def send(self, obj):
        return self.conn.send(obj)

    def run(self, conn):
        pass
