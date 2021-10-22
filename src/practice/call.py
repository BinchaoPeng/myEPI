class Call:
    def __init__(self, name):
        self.name = name

    def call_name(self):
        print("hello", self.name)
        return self.name

    def __call__(self, *args, **kwargs):
        print("__call__")
        return self.call_name()

if __name__ == '__main__':
    """
    类似于在类中重载 () 运算符
    通过在 Call 类中实现 __call__() 方法，使的 call 实例对象变为了可调用对象。
    """
    call = Call("PBC")
    name = call.call_name()
    print(name)
    print(type(call))
    b = call()
