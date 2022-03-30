import sys
import time
import types
from multiprocessing.managers import NamespaceProxy


def Proxy(target, moduel=None):
    dic = {
        'types': types,
        'time': time}
    exec('''def __getattr__(self, key):
        if key[0] == '_':
            return object.__getattribute__(self, key)
        result = self._callmethod('__getattribute__', (key,))
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(key, args, kwargs)
            return wrapper
        return result''', dic)
    proxyName = target.__name__ + "Proxy"
    ProxyType = type(proxyName, (NamespaceProxy,), dic)
    ProxyType._exposed_ = tuple(dir(target))
    setattr(sys.modules[__name__], ProxyType.__name__, ProxyType)
    return ProxyType
