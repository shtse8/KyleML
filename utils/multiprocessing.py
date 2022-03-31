import sys
import time
import types
from multiprocessing.managers import NamespaceProxy


def Proxy(target, module=None):
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
    proxy_name = target.__name__ + "Proxy"
    proxy_type = type(proxy_name, (NamespaceProxy,), dic)
    proxy_type._exposed_ = tuple(dir(target))
    setattr(sys.modules[__name__], proxy_type.__name__, proxy_type)
    return proxy_type
