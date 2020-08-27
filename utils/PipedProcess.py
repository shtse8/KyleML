from .BufferConnection import BufferConnection
import torch.multiprocessing as mp


class Process:
    def __init__(self):
        self.process = None
        self.started = False
        self.args = ()

    def start(self):
        if self.started:
            raise Exception("Process is started")

        self.started = True
        self.process = mp.Process(target=self.run, args=self.args)
        self.process.start()
        return self

    def run(self):
        pass


class PipedProcess(Process):
    def __init__(self):
        super().__init__()

    def start(self):
        connections = mp.Pipe(True)
        self.conn = BufferConnection(connections[0])
        child_conn = BufferConnection(connections[1])
        self.args = (child_conn,)
        return super().start()

    def poll(self):
        return self.conn.poll()

    def recv(self):
        return self.conn.recv()

    def send(self, obj):
        return self.conn.send(obj)

    def run(self, conn):
        pass
