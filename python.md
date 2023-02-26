

# 目录

[TOC]

# 基本数据类型

## 列表

### 列表推导式

**[表达式 for 变量 in 序列或迭代对象 if条件]**

for可以有两个来构成循环,详见[product](#product)

```python
# if no annotation, use python 3
# 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
def letterCombinations(digits: str) -> list:
    KEY = {'2': ['a', 'b', 'c'],
           '3': ['d', 'e', 'f'],
           '4': ['g', 'h', 'i'],
           '5': ['j', 'k', 'l'],
           '6': ['m', 'n', 'o'],
           '7': ['p', 'q', 'r', 's'],
           '8': ['t', 'u', 'v'],
           '9': ['w', 'x', 'y', 'z']}
    if digits == '':
        return []
    ans = ['']
    for num in digits:
        ans = [pre + suf for pre in ans for suf in KEY[num]]
    return ans
```

此外亦有字典推导式，集合推导式

## 字典

字典在python中非常常用，可以当作哈希表

创建空字典:

```python
c = dict() # 或者
c1 = {}
```

注意:{} is not None

获取字典的键，值

```python
c.keys()  # 此时的类型为dict_keys，可以用list()等进行转换,python2返回的是list
c.values()
```

字典也可以使用pop()，但必须给出key

### OrderedDict

使用OrderedDict会根据放入元素的先后顺序进行排序。所以输出的值是排好序的

需要包含模块collections

### 更新字典

```python
dict.update(dict2)
```

dict2 -- 添加到指定字典dict里的字典

## 集合

集合（set）是一个无序的不重复元素序列

可以使用大括号 **{ }** 或者 **set()** 函数创建集合，注意：创建一个空集合必须用 **set()** 而不是 **{ }**，因为 **{ }** 是用来创建一个空字典。

# 语句

`for/while
{...[break]}
else`

若自然结束循环，else部分执行;若break,则不执行

# 语法解析

1.  a,b = b,a是怎么实现的

**在python中，会在过程中生成一个元组 c，并且c = (b ,a)，然后进行a = c[0] ， b = c[1] 的操作**

2. a,b=b,a+b

先计算=右边的值

## 不定参数

将不定数量的参数传递给一个函数

```python
def fun(name, *args):
    # print(f'type:{type(args)}' )  #此时args为tuple
    print(f'你好,{name}')
    for i in args:
        print(f'你的宠物有:{i}')
fun('Bob', 'cat', 'dog', 'bird')
def g(**kwargs):
    print(kwargs)  # type: dict
g(expected_type_list='InvalidField')
```

本质是将*/**后面的可迭代对象解包出来

## **DocStrings**

文档字符串是一个重要工具，用于解释文档程序，帮助你的程序文档更加简单易懂。

``````python
def function():
        ''' say something here！
        '''
        pass

print (function.__doc__) # 调用 doc    ->say something here！
``````



# 内建函数

实用内建函数

## join

连接字符串

一般用来连接可迭代对象的各个元素

```python
list=['1','2','3','4','5']
print(''.join(list)) # 12345
```

## format

```python
print('{:.2%}'.format(0.6667)
```

输出百分比，保留两位小数

‘:’用来指定各种格式

```python
print("------Script: {:<3}.py ------".format('sa'))
# 长度大于等于3，否则以空白补齐
print("{:x}".format(16)) # 16进制
"""
^, <, > 分别是居中、左对齐、右对齐，后面带宽度， : 号后面带填充的字符，只能是一个字符，不指定则默认是用空格填充。
"""
```

不仅仅用于打印,如:

```python
CMD = r"sudo setpci -s {pit} {pil}"   # 加r防转义
pit = 3
pil = 1
rdc = CMD.format(pit=pit,pil=pil)  # sudo setpci -s 3 1
```

## 数字相关

### float

```python
print(float(num))  # 3.6e-3 和'3.6e-3'都可以
```

返回一个**十进制浮点型数值（小数）**。

float()括号内可以是三种类型的数据：
1.二进制、八进制、十进制、十六进制的**整数**。`1,0xff,0b111,0o10`
2.bool（布尔值True和False）。
3.表示**十进制**数字的字符串,范围较大。`'32','4e5','.4'...`

特别的，`'inf'`或者`'Infinity'`表示无穷大

### hex

```python
print(int(num))   # 转成10进制,等同于print(num)
print(hex(num))  # 转成16进制
print(bin(num))  # 转成2进制
# num遵循python数字表示法即可
# int还可以将字符串转换成整型:
int('0xA',0)  # 10
```

## enumerate

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

```python
for i, ch in enumerate(s):
    # ...
```

## sorted()

```python
q = sorted(l) # 排序，返回新列表
# 对字符串操作时
s = 'dca'
print(''.join(sorted(s)))  # ->acd
```

## sort()

``````python
l.sort() # 对原列表进行操作
``````

## zip

将可迭代的对象作为参数，将对象中对应的元素打包成一个个__元组__，然后返回由这些元组组成的对象.

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同

可以利用zip将矩阵对角线的数字互换:(按列从新排)

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix[:] = zip(*matrix)
print(matrix)   # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

## map

```python
map(function, iterable, ...) # function -- 函数 iterable -- 一个或多个序列
```

在上面例子中，matrix中元素成了元祖，可以利用map函数;map返回是迭代器，需要用其他函数转换

```python
matrix[:] = map(list, zip(*matrix))  # [:]有隐形转换
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```

## filter

用法与map类似，也返回迭代器

例子:

```python
q = filter(lambda x: x % 2 == 1, [1, 2, 3, 4, 5])  #  过滤出奇数
```

## range

返回类型range object,可迭代

r = range(5) means a object like (0,1,2,3,4)

r[0] is valid

## 类相关

### setattr

用于设置属性值，该属性不一定是存在的

### hasattr

用于判断对象是否包含对应的属性，有该属性返回 True，否则返回 False。

由`getattr`和exception实现

```python
class A(object):
    b = 6


def main():
    a = A()
    setattr(a, 'c', 7)
    a.d = 8
    print a.c, a.d  # 7 8
    print hasattr(a, 'd') # True
```

## dir

**dir()** 函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；带参数时，返回参数的属性、方法列表

## eval

执行一个字符串表达式，并返回表达式的值

eval() 函数也可以直接用来提取用户输入的多个值。

例如：

```python
a,b=eval(input())
```

输入：**10,5**，得到 **a=10，b=5**。

## exec

执行储存在字符串或文件中的 Python 语句

[返回目录](#目录)

# 命名空间

**命名空间是对变量名的分组划分**。

LEGB规定了查找一个名称的顺序为：local-->enclosing function locals-->global-->builtin

通过把一些模块进行加载和重命名(库中的类)操作，可以加快查找速度

需要定义class`LibLoader`,实现`_get_library_instance`

在基类进行lib_loader,子类就可以invoke，如:

`self.sec = self.lib.security`

# 常用模块

## os

**os** 模块提供了非常丰富的方法用来处理文件和目录

```python
system(command:str) # 在一个子shell执行command命令
```

### os.path

跟路径相关

`os.path.abspath(__file__)`代表当前脚本的绝对路径

## collections

### defaultdict

defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值

```python
dict =defaultdict( factory_function)
```

factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值，比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0

### Counter

对可迭代对象的各个元素进行计数，统计

```python
s = 'aabbcc'
C = Counter(s) # 此时类型为Counter类，类似于字典
```



## typing

方法参数的类型检查

## itertools

The module standardizes a core set of fast, memory efficient tools that are useful by themselves or in combination.

### permutations

返回可迭代对象的排列，默认全排列

```python
list(permutations(range(3), 2)) --> [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
```

### product

用于求多个可迭代对象的笛卡尔积(Cartesian Product)，它跟嵌套的 for 循环等价，即各个元素排列组合

内部生成元素为tuple

## functools

### reduce

对参数序列中元素进行累积。

```python
# 计算阶乘
def fab(n: int):
    return reduce(lambda x, y: x * y, range(1, n + 1))
```

### wraps

Python[__装饰器__](#装饰器)（decorator）在实现的时候，被装饰后的函数其实已经是另外一个函数了（函数名等函数属性会发生改变），为了不影响，Python的functools包中提供了一个叫wraps的decorator来消除这样的副作用。写一个decorator的时候，最好在实现之前加上functools的wrap，它能保留原有函数的名称和[docstring](#DocStrings)。

## threading

线程模块

本模块定义了thread的类

创建线程，即从thread继承:

```python
t = threading.Thread(group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None)
# 一般用到target，即自定义的某个函数；args,函数的入参
```

线程运行：

```python
t.start()
```

### daemon

设置线程为守护线程

```python
t.setDaemon(True)  # 设置必须要在start之前
```

主线程为非守护线程(前台线程)，当前台线程结束时，守护线程也会结束

## multiprocessing

多进程

```python
import multiprocessing
import os
def saku1():
    print('saku1 is called,pid is {}'.format(os.getpid()))
p2 = multiprocessing.Process(target=saku1)   
p2.start()  # 不能直接用run，是假的多进程，几个函数的pid会相同
```

## subprocess

Popen 是 subprocess的核心，子进程的创建和管理都靠它处理

```python
import subprocess
cmd = 'ls'  
subprocess.Popen(cmd, shell = True)
```

该模块与try, shell的组合用法:

```python
try:
    import xlrd
except ImportError:
    import subprocess
    subprocess.check_output('sudo -E pip3 install xlrd -U', shell=True)
    # subprocess.check_output('sudo /opt/python32/bin/pip install xlrd -U', shell=True)
    import xlrd
```

## queue

queue 模块即队列，特别适合处理信息在多个线程间安全交换的多线程程序中

### 1 Queue(maxsize=0)

先进先出(First In First Out: FIFO)队列，最早进入队列的数据拥有出队列的优先权

入参 maxsize 是一个整数，用于设置队列的最大长度。一旦队列达到上限，插入数据将会被阻塞，直到有数据出队列之后才可以继续插入。如果 maxsize 设置为小于或等于零，则队列的长度没有限制。

示例如下：

```python
import queue
q = queue.Queue()  # 创建 Queue 队列
for i in range(3):
    q.put(i)  # 在队列中依次插入0、1、2元素
for i in range(3):
    print(q.get())  # 依次从队列中取出插入的元素，数据元素输出顺序为0、1、2
```

### 2 LifoQueue(maxsize=0)

后进先出(Last In First Out: LIFO)队列，最后进入队列的数据拥有出队列的优先权，就像栈一样。

入参 maxsize 与先进先出队列的定义一样。

示例如下：

```python
import queue
q = queue.LifoQueue()  # 创建 LifoQueue 队列
for i in range(3):
    q.put(i)  # 在队列中依次插入0、1、2元素
for i in range(3):
    print(q.get())  # 依次从队列中取出插入的元素，数据元素输出顺序为2、1、0
```

### 3 PriorityQueue(maxsize=0)

优先级队列，比较队列中每个数据的大小，__值最小的数据__拥有出队列的优先权。数据一般以元组的形式插入，典型形式为(priority_number, data)。如果队列中的数据没有可比性，那么数据将被包装在一个类中，忽略数据值，仅仅比较优先级数字。

示例如下：

```python
import queue
q = queue.PriorityQueue()  # 创建 PriorityQueue 队列
data1 = (1, 'python')
data2 = (2, '-')
data3 = (3, '100')
style = (data2, data3, data1)
for i in style:
    q.put(i)  # 在队列中依次插入元素 data2、data3、data1
for i in range(3):
    print(q.get())  # 依次从队列中取出插入的元素，数据元素输出顺序为 data1、data2、data3
```

获取队列元素个数

`q.qsize()`

判断队列是否为空

`q.empty()`

## configparser

对配置文件(ini，cfg)进行操作的模块

示例:

```python
"""
test.ini
[GLOBAL]
CON_NUM=1

[CON_000]
GAPP_VERSION=3
GAPP_APP_TYPE=2
GAPP_LOCAL_SSTY=90
GAPP_LOCAL_LOGICAL_ID=3
GAPP_LOCAL_SSID=3
GAPP_REMOTE_SSTY=30
"""
class MyParser(configparser.ConfigParser):
    # 重写optionform，防止大小写被改变
    def optionxform(self, optionstr):
        return optionstr


c = MyParser()
path = 'test.ini'
c.read(path)
c.set('CON_000', 'GAPP_VERSION', '5')   # 将此字段改成5
with open(path, 'w+') as f:
    c.write(f, space_around_delimiters=False)  # 改完需要写入，False表示‘=’左右无空格
```

## logging



```python
import logging

logger = logging.getLogger('Test_Device')  # 设置logger名称，没有设置则是root
logger.setLevel(level=logging.INFO)  # 设置默认的日志级别,DEBUG才会全部输出
handler = logging.FileHandler("0827.txt", mode='w')  # 设置文件处理的文件名，默认mode:'a',追加
# 创建日志格式对象
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)  # logger日志对象加载FileHandler对象
logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")
logger.critical("炸了")
```

## struct

用来在c语言中的结构体与python中的字符串之间进行转换，数据一般来自文件或者网络

格式字符串的第一个字符可用于指示打包数据的字节顺序，大小和对齐方式：

| 字符 | 字节顺序      | 大小     | 对齐方式 |
| ---- | ------------- | -------- | -------- |
| `@`  | 按原字节      | 按原字节 | 按原字节 |
| `=`  | 按原字节      | 标准     | 无       |
| `<`  | 小端          | 标准     | 无       |
| `>`  | 大端          | 标准     | 无       |
| `!`  | 网络（=大端） | 标准     | 无       |

如果第一个字符不是其中之一，则假定为 `'@'` 。

| 格式 | C 类型               | Python 类型       | 标准大小 | 注释     |
| ---- | -------------------- | ----------------- | -------- | -------- |
| `x`  | 填充字节             | 无                |          |          |
| `c`  | `char`               | 长度为 1 的字节串 | 1        |          |
| `b`  | `signed char`        | 整数              | 1        | (1), (2) |
| `B`  | `unsigned char`      | 整数              | 1        | (2)      |
| `?`  | `_Bool`              | bool              | 1        | (1)      |
| `h`  | `short`              | 整数              | 2        | (2)      |
| `H`  | `unsigned short`     | 整数              | 2        | (2)      |
| `i`  | `int`                | 整数              | 4        | (2)      |
| `I`  | `unsigned int`       | 整数              | 4        | (2)      |
| `l`  | `long`               | 整数              | 4        | (2)      |
| `L`  | `unsigned long`      | 整数              | 4        | (2)      |
| `q`  | `long long`          | 整数              | 8        | (2)      |
| `Q`  | `unsigned long long` | 整数              | 8        | (2)      |
| `n`  | `ssize_t`            | 整数              |          | (3)      |
| `N`  | `size_t`             | 整数              |          | (3)      |
| `e`  | (6)                  | float             | 2        | (4)      |
| `f`  | `float`              | float             | 4        | (4)      |
| `d`  | `double`             | float             | 8        | (4)      |
| `s`  | `char[]`             | 字节串            |          |          |
| `p`  | `char[]`             | 字节串            |          |          |
| `P`  | `void *`             | 整数              |          | (5)      |

在 3.3 版更改: 增加了对 `'n'` 和 `'N'` 格式的支持

在 3.6 版更改: 添加了对 `'e'` 格式的支持。

```python
import struct

values = (1, b'abc', 2.7)
s = struct.Struct('I3sf')  
'''
I:unsigned int
s:char[]
f:float
'''
p = s.pack(*values)
u = s.unpack(p)

print(s.size)
print(u)
#12 会有字节对齐
#(1, b'abc', 2.700000047683716)

#python 2
def main():
    s = struct.pack('<I', 0xFFAA)
    s = struct.unpack('<I', s)
    print "0x{:X}".format(s[0])  # 如果指定大小端，就都指定，不然就可能会发生错误
```

## gc

python里`gc.collect()`命令可以回收没有被使用的空间，但是这个命令还会返回一个数值，是清除掉的垃圾变量的个数

## random

```python
import random


def main():
    l = [1, 2, 3, 4, 5]
    random.shuffle(l)  # 将l中的元素打乱
    print(l)
```

```python
sample(population,k)
# Chooses k unique random elements from a population sequence
```

[↑top](#目录)

## timeit

更好的时间精度模块

`default_timer()`根据不同的操作系统和python版本，选择最为合适的计时器。

## platform

用于查看当前操作系统的信息，来采集系统版本位数计算机类型名称内核等一系列信息

```python
platform.architecture() #获取操作系统的位数，('32bit', 'ELF')
platform.platform()    #获取操作系统名称及版本号，'Linux-3.13.0-46-generic-i686-with-Deepin-2014.2-trusty'
```

## string

该模块存储一些常用的变量整体,如:`string.letters`存储了各种字母,`digits`存储了数字,`punctuation`存储了符号

## binascii

包含很多在二进制和二进制表示的各种ASCII码之间转换的方法

`binascii.hexlify(data)`

返回__二进制__数据 *data* 的十六进制表示形式。在py3中data如果为str会报错,需要encode，转化了二进制(ascii码)

## argparse

`argparse`是一个用来解析命令行参数的 Python 库

## ctypes

提供了与 C 兼容的数据类型，并允许调用 DLL 或共享库中的函数

使用的例子

```python
class Unel(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('poh', ctypes.c_uint16),
        ('ch', ctypes.c_uint8),
    ]


p = Unel(12, 8)
print(p.__getattribute__('poh'))
print(p.poh)
print(p._fields_)
```

例子2

```python
import ctypes

buf = ctypes.create_string_buffer(32)
pattern = [0xaa, 0xbb, 0xcc, 0xdd]
meta_buf = bytearray(pattern)
address = ctypes.addressof(buf)
ctypes.memmove(address, bytes(meta_buf), len(meta_buf))
print(buf.raw)

```

# 文件

文件的读写操作默认使用系统编码，可以通过调用 `sys.getdefaultencoding()` 来得到。 在大多数机器上面都是utf-8编码。如果你已经知道你要读写的文本是其他编码方式， 那么可以通过传递一个可选的 `encoding` 参数给open()函数。如下所示：

```python
with open('somefile.txt', 'rt', encoding='latin-1') as f:
    ...
```

Python支持非常多的文本编码。几个常见的编码是ascii, latin-1, utf-8和utf-16。 在web应用程序中通常都使用的是UTF-8。 ascii对应从U+0000到U+007F范围内的7位字符。 latin-1是字节0-255到U+0000至U+00FF范围内Unicode字符的直接映射。 **当读取一个未知编码的文本时使用latin-1编码永远不会产生解码错误。**

refer: https://python3-cookbook.readthedocs.io/zh_CN/latest/c05/p01_read_write_text_data.html

# 生成器

节省内存空间，不会一次性生成所有的数据，而是什么时候需要，什么时候生成

```python
from itertools import product

test_pract = [0, 1]
test_pit = [0, 1, 2, 3]
test_meset = [0, 1]


def generate_test_param():
    for pract, pit, mset in product(test_pract, test_pit, test_meset):
        yield pract, pit, mset


for ele in generate_test_param():
    print('now tested param: {} {} {}'.format(ele[0], ele[1], ele[2]))

```

# 装饰器

在不改变原有功能代码的基础上,添加额外的功能,如用户验证等。有助于让代码更简短

```python
def timefn(fn):
    """计算性能的修饰器"""

    @wraps(fn)  # wraps本身也是一个装饰器，它能把原函数的元信息拷贝到装饰器函数中，这使得装饰器函数也有和原函数一样的元信息
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")  # print(f...)为格式化输出用法
        return result

    return measure_time
@timefn
def fun(argc):
    pass
```

使用类装饰器还可以依靠类内部的 `__call__`方法

```python
class DivideDec(object):
    """
    判断除法是否valid
    """
    def __init__(self, func):  # 将divide作为参数传给DivideDec
        self.func = func

    def __call__(self, *args, **kwargs):
        exp_res = kwargs.get('exp', 'success')
        try:
            res = self.func(*args, **kwargs)
        except Exception as err:
            if exp_res in err.args[0]:
                print('devide as expected:{}'.format(err.args[0]))
            else:
                raise Exception('expected is {} but actual is {}'.format(exp_res, err.args[0]))
        else:
            return res


@DivideDec
def divide(a, b, **kwargs):
    return a/b


divide(3, 0, exp='division by zero')
```

# 异常处理

```python
try:
    4 / 0
except Exception as e: # 将捕获到的异常对象赋值给e
    print(e) # 访问异常详细信息
else:
    # 当 try 块没有出现异常时，程序会执行 else 块; else块可以不要
finally:
    pass
    # finally中的语句一定会被执行
```

assert:用于判断一个表达式，在表达式条件为 false 的时候触发异常。

例如`assert 0`就会触发异常

# 类

## 继承

```python
import queue


class BaseDevice():  # subclassed by RSSP1_DEV
    __deviceName = None
    __deviceId = None  # cannot visit/inherit
    maxQueueSize = 1024  # inherited by RSSP1_DEV
    TimeScenario = None

    def __init__(self, name, id):  # overridden in RSSP1_DEV
        self.__deviceName = name
        self.__deviceId = id
        self.inQ = queue.Queue(self.maxQueueSize)
        self.TimeScenario = []

    def get_device_id(self):
        print('id is 0x%x' % self.__deviceId)


class RSSP1_DEV(BaseDevice):
    cycleMsgID = 99
    __log = None

    def __init__(self, name, id):
        BaseDevice.__init__(self, name, id)


class RAW_DEV(BaseDevice):
    apptype = 3

    def __init__(self, name, id):
        super().__init__(name, id)  # 在单类继承中,不需要父类的名称来调用父类的函数(super用来调父类)


r = RSSP1_DEV('rssp', 0xA)
r.get_device_id()
print(r.maxQueueSize)
ra = RAW_DEV('raw', 2001)
ra.get_device_id()
```
## property
```python
class BaseDevice():
    _deviceName: str = None  # 保护变量，只能自己或子类使用
    __deviceId = None # 私有变量，只能内部使用

    def __init__(self, name, id):
        self._deviceName = name
        self.__deviceId = id

    @property
    def device_name(self):
        return self._deviceName

    @property
    def deviceID(self):
        return self.__deviceId


if __name__ == '__main__':
    b = BaseDevice('raw', 0xaa)
    print(b.device_name)
    print(b.deviceID)
```

property使用场景：1.修饰方法，使方法可以像属性一样访问。调用时不用括号，**不能带参数**

2.与所定义的属性配合使用，这样可以防止属性被修改

property相当于getter，还可以用setter装饰器，对setter进行设置

```python
# python 2.7
class Geeks(object):
    def __init__(self):
        self._age = 0

    # using property decorator
    # a getter function
    @property
    def age(self):
        print("getter method called")
        return self._age

    # a setter function
    @age.setter
    def age(self, a):
        if (a < 18):
            raise ValueError("Sorry you age is below eligibility criteria")
        print("setter method called")
        self._age = a
```

## 嵌套类 nested class

嵌套类（也叫内类）是在另一个类中定义的。它并不能提高执行时间，但可以通过将相关的类归为一组来帮助程序的可读性和维护，而且还可以将嵌套类隐藏起来，不让外界看到。

```python
class Dept:
    def __init__(self, dname):
        self.dname = dname

    class Prof:
        def __init__(self, pname):
            self.pname = pname

math = Dept("Mathematics")
mathprof = Dept.Prof("Mark")

print(math.dname)
print(mathprof.pname)
'''
out:
Mathematics
Mark
'''
```

## descriptor

如果一个新式类定义了`__get__`, `__set__`,` __delete__`方法中的一个或者多个，那么称之为descriptor

* Dynamic lookups

```python
# python 2.7
import os


class DirectorySize(object):
    def __get__(self, obj, objtype=None):
        return len(os.listdir(obj.dirname))


class Directory:
    size = DirectorySize()  # Descriptor instance

    def __init__(self, dirname):
        self.dirname = dirname  # Regular instance attribute


def main():
    s = Directory('song')
    print s.size
    g = Directory('game')
    print g.size


if __name__ == '__main__':
    main()
#  File count is automatically updated according to folder and file numbers
```

## override

如果需要子类使用某方法必须重新自己实现时，子类override，父类使用exception

```python
# python 2.7


class A(object):
    def test(self):
        """
        Main test method that should be overridden by child classes.
        """
        raise NotImplementedError("Test function needs overridden for each test.")


class B(A):
    def test(self):
        # pass
        print 'yes i have implemented this method'


def main():
    b = B()

    b.test()
# 如果B未实现test，则报错

if __name__ == '__main__':
    main()
```

## 类的特殊函数

```python
class Target:
    def __init__(self, name):
        self.__id = name

    def __getitem__(self, item):
        """make class[item] usable"""
        if item >= len(self.__id):
            raise RuntimeError('invalid input')
        return self.__id[item]

    def __str__(self):
        """
        override when call class
        """
        if not isinstance(self.__id, str):
            raise RuntimeError('invalid input')
        return self.__id + ' sacred'


a = Target('Cute')
print(a)
print(a[3])

```

# 算法

## 递归

### 中序遍历

``````python
class Solution: # 递归通用写法
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(cur):
            if not cur:
                return
            dfs(cur.left)
            res.append(cur.val)
            dfs(cur.right)
        res = []
        dfs(root)
        return res
``````

### 快排

```python
def qsort(nums: List[int]) -> List[int]:
    if len(nums) <= 1: # 指定结束条件
        return nums
    left = [x for x in nums[1:] if x <= nums[0]]
    right = [x for x in nums[1:] if x > nums[0]]
    return qsort(left) + [nums[0]] + qsort(right)
```
## 回溯

回溯算法实际上一个类似枚举的搜索尝试过程，主要是在搜索尝试过程中寻找问题的解，当发现已不满足求解条件时，就“回溯”返回，尝试别的路径。回溯也是用递归实现的
```python
'''

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的数字可以无限制重复被选取。
说明：
所有数字（包括 target）都是正整数。
解集不能包含重复的组合。
来源：力扣（LeetCode） 39
'''
def combinationSum(candidates, target):
    res = []

    def helper(nums, target, res_list):
        if target < 0:
            return
        if target == 0:
            res.append(res_list)
        for i, c in enumerate(nums):
            helper(nums[i:], target - c, res_list + [c]) # 解集不能包含重复的组合

    helper(candidates, target, [])
    return res
```

## 二分查找

```python
# 必须按关键字大小有序排列
def bin_search(nums, val):
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == val:
            return mid
        elif nums[mid] < val:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

## 贪心

贪心算法（又称贪婪算法）是指，在对[问题求解](https://baike.baidu.com/item/问题求解/6693186)时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，[算法](https://baike.baidu.com/item/算法/209025)得到的是在某种意义上的局部最优解

```python
''' 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

如果某一个作为 起跳点 的格子可以跳跃的距离是 3，那么表示后面 3 个格子都可以作为 起跳点。
可以对每一个能作为 起跳点 的格子都尝试跳一次，把 能跳到最远的距离 不断更新。
如果可以一直跳到最后，就成功了。
'''
def canJump(nums: List[int]) -> bool:
    k = 0  # k：维护能跳到的最远距离
    for i, j in enumerate(nums):
        if i > k:
            return False
        k = max(k, i + j)
    return True
```

# 开发工具

## 2to3

2to3 是一个 Python 程序，它可以用来读取 Python 2.x 版本的代码，并使用一系列的 *修复器* 来将其转换为合法的 Python 3.x 代码。详见[2to3 - 自动将 Python 2 代码转为 Python 3 代码 — Python 3.7.12 文档](https://docs.python.org/zh-cn/3.7/library/2to3.html)

## pylint

代码规范检查

## pdb

代码调试工具

```python
#进入pdb
import pdb
pdb.set_trace()
```

可以打印各种变量或者表达式的值

按u到上一层

## pycharm

常用的python开发IDE

tools旁边的vcs可以开启git等UI

views→appearance→status bar widgets控制想显示的组件

## anaconda
用于python的虚拟环境控制，同时自带很多实用module
自带虚拟环境base
```sh
conda env list  # 查看已有的虚拟环境
conda create --name my_test python=3.7   #创建一个名称为my_test的虚拟环境，python版本3.7，也可以用pycharm的UI进行操作
conda activate my_test  # 切换到某个虚拟环境; 可以从anaconda的navigator的UI进入
```

[↑top](#目录)
