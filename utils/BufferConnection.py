import collections
import time

class BufferConnection:
    def __init__(self, connection):
        self.connection = connection
        self.buffer = collections.deque(maxlen=1000)

    def _bufferAll(self):
        while self.connection.poll():
            self.buffer.append(self.connection.recv())

    def hasMessage(self):
        return len(self.buffer) > 0

    def poll(self):
        self._bufferAll()
        return self.hasMessage()

    def recv(self):
        # blocking
        while True:
            self._bufferAll()
            if self.hasMessage():
                break
            time.sleep(0.01)
        return self.buffer.popleft()

    def send(self, obj):
        return self.connection.send(obj)