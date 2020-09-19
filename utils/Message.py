class Message:
    def __init__(self):
        pass


class NetworkInfo(Message):
    def __init__(self, stateDict, version):
        self.stateDict = stateDict
        self.version = version


class LearnReport(Message):
    def __init__(self, loss=0, steps=0, drops=0):
        self.loss = loss
        self.steps = steps
        self.drops = drops


class EnvReport(Message):
    def __init__(self):
        self.rewards = 0

class MethodCallRequest(Message):
    def __init__(self, method, args):
        self.method = method
        self.args = args


class MethodCallResult(Message):
    def __init__(self, result):
        self.result = result
