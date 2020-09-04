from multiprocessing.managers import NamespaceProxy
import types
import sys
import time

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
                tic = time.perf_counter()
                self._callmethod(key, args)
                if key == "add":
                    print("p", time.perf_counter() - tic)
            return wrapper
        return result''', dic)
    print(dic)
    proxyName = target.__name__ + "Proxy"
    ProxyType = type(proxyName, (NamespaceProxy,), dic)
    ProxyType._exposed_ = tuple(dir(target))
    setattr(sys.modules[__name__], ProxyType.__name__, ProxyType)
    return ProxyType
