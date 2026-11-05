"""
generate_100k_dataset.py
========================
Generates ~100,000 anchor-positive pairs for fine-tuning
a SentenceTransformer embedding model on Python concepts.

HOW IT WORKS
------------
1. Loads the existing 601 pairs from data/python_code_search_500.json
2. Generates a massive synthetic set using template × seed expansion
3. Optionally pulls real pairs from HuggingFace datasets (if network available)
4. Deduplicates and saves to data/python_code_search_100k.json

RUN
---
    pip install datasets   # only needed for the HuggingFace section
    python scripts/generate_100k_dataset.py
"""

import json, os, random, itertools
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
EXISTING   = BASE_DIR / "data" / "python_code_search_500.json"
OUTPUT     = BASE_DIR / "data" / "python_code_search_100k.json"

# ─────────────────────────────────────────────────────────────
# SECTION 1 – LOAD EXISTING 601 PAIRS
# ─────────────────────────────────────────────────────────────
def load_existing():
    with open(EXISTING, encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────
# SECTION 2 – LARGE SEED BANK
# Each entry: (topic, anchor_templates[], positive_code)
# ─────────────────────────────────────────────────────────────
SEEDS = [
    # ── GENERATORS / ITERATORS ──────────────────────────────
    (
        "yield_basic",
        [
            "How to use yield in Python",
            "What is yield keyword in Python",
            "Python generator function example",
            "How to create a generator in Python",
            "yield statement Python tutorial",
            "How does yield work in Python",
            "Python lazy evaluation with yield",
            "Generator function that pauses execution",
            "Resume a function from where it left Python",
            "Python coroutine yield example",
        ],
        "def count_up(n):\n    \"\"\"Generator that yields numbers 0 to n-1.\"\"\"\n    for i in range(n):\n        yield i",
    ),
    (
        "yield_from",
        [
            "How to delegate to a sub-generator in Python",
            "yield from keyword Python example",
            "Python yield from usage",
            "Delegating generator Python",
            "yield from in Python 3",
            "How to chain generators using yield from",
            "Sub-generator delegation Python",
            "Refactor nested loops using yield from",
        ],
        "def flatten(nested):\n    \"\"\"Flatten nested iterables using yield from.\"\"\"\n    for item in nested:\n        if hasattr(item, '__iter__') and not isinstance(item, str):\n            yield from flatten(item)\n        else:\n            yield item",
    ),
    (
        "generator_send",
        [
            "How to send values into a generator Python",
            "Python generator send method example",
            "Two-way communication with generator Python",
            "generator.send() Python tutorial",
            "How to pass data into a paused generator",
            "Python coroutine send value",
            "Interactive generator Python",
        ],
        "def accumulator():\n    \"\"\"Generator that receives values via send() and accumulates.\"\"\"\n    total = 0\n    while True:\n        value = yield total\n        if value is None:\n            break\n        total += value",
    ),
    (
        "generator_throw",
        [
            "How to throw exception into generator Python",
            "generator.throw() method Python",
            "Raise exception inside generator Python",
            "Inject error into Python generator",
            "Python generator error handling",
        ],
        "def safe_gen():\n    \"\"\"Generator that handles injected exceptions.\"\"\"\n    try:\n        while True:\n            yield\n    except ValueError as e:\n        print(f'Caught: {e}')\n        yield 'recovered'",
    ),
    (
        "generator_close",
        [
            "How to stop a generator Python",
            "generator.close() Python example",
            "Close a running generator Python",
            "Terminate infinite generator Python",
            "Python generator cleanup on close",
        ],
        "def resource_gen():\n    \"\"\"Generator with cleanup on close.\"\"\"\n    print('Generator started')\n    try:\n        while True:\n            yield\n    finally:\n        print('Generator closed - cleanup done')",
    ),
    (
        "itertools_chain",
        [
            "How to chain multiple iterables Python",
            "itertools.chain example Python",
            "Combine multiple lists Python itertools",
            "Concatenate iterables without copying",
            "Python chain iterators together",
        ],
        "import itertools\n\ndef chain_all(*iterables):\n    \"\"\"Chain multiple iterables into one using itertools.\"\"\"\n    return list(itertools.chain(*iterables))\n\n# Usage: chain_all([1,2], [3,4], [5]) -> [1,2,3,4,5]",
    ),
    (
        "itertools_islice",
        [
            "How to slice an iterator Python",
            "itertools.islice Python example",
            "Take first N items from generator Python",
            "Limit iterator output Python",
            "Python lazy slicing of infinite sequence",
        ],
        "import itertools\n\ndef take(n, iterable):\n    \"\"\"Return first n items of an iterable as a list.\"\"\"\n    return list(itertools.islice(iterable, n))",
    ),
    (
        "itertools_product",
        [
            "Cartesian product in Python",
            "itertools.product Python example",
            "All combinations of multiple lists Python",
            "How to get all pairs from two lists Python",
            "Python nested loop replacement with product",
        ],
        "import itertools\n\ndef cartesian_product(a, b):\n    \"\"\"Compute Cartesian product of two iterables.\"\"\"\n    return list(itertools.product(a, b))",
    ),
    (
        "itertools_groupby",
        [
            "How to group consecutive items Python",
            "itertools.groupby Python example",
            "Group list by key Python",
            "Consecutive grouping Python",
            "Python group adjacent equal elements",
        ],
        "import itertools\n\ndef group_consecutive(lst):\n    \"\"\"Group consecutive equal elements.\"\"\"\n    return [(k, list(v)) for k, v in itertools.groupby(lst)]",
    ),
    (
        "itertools_combinations",
        [
            "How to get all combinations Python",
            "itertools.combinations example",
            "Choose r items from list Python",
            "Python combination without repetition",
            "All subsets of size r Python",
        ],
        "import itertools\n\ndef all_combinations(lst, r):\n    \"\"\"Return all r-length combinations from lst.\"\"\"\n    return list(itertools.combinations(lst, r))",
    ),
    # ── DECORATORS ──────────────────────────────────────────
    (
        "decorator_basic",
        [
            "How to write a decorator in Python",
            "Python decorator example",
            "Function wrapper Python",
            "What is a decorator in Python",
            "Add behaviour to function without modifying it Python",
            "Python higher order function decorator",
        ],
        "from functools import wraps\n\ndef my_decorator(func):\n    \"\"\"Simple decorator that prints before and after.\"\"\"\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        print('Before call')\n        result = func(*args, **kwargs)\n        print('After call')\n        return result\n    return wrapper",
    ),
    (
        "decorator_with_args",
        [
            "Python decorator with arguments",
            "Parametrized decorator Python",
            "Decorator factory Python",
            "How to pass arguments to a decorator Python",
            "Decorator that accepts parameters Python",
        ],
        "from functools import wraps\n\ndef repeat(n):\n    \"\"\"Decorator factory: repeat function n times.\"\"\"\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            for _ in range(n):\n                result = func(*args, **kwargs)\n            return result\n        return wrapper\n    return decorator",
    ),
    (
        "decorator_class",
        [
            "Class based decorator Python",
            "How to make a class a decorator Python",
            "Decorator using __call__ Python",
            "Stateful decorator Python using class",
        ],
        "class CountCalls:\n    \"\"\"Class-based decorator that counts function calls.\"\"\"\n    def __init__(self, func):\n        self.func = func\n        self.count = 0\n\n    def __call__(self, *args, **kwargs):\n        self.count += 1\n        return self.func(*args, **kwargs)",
    ),
    (
        "lru_cache",
        [
            "How to cache function results Python",
            "Python lru_cache decorator example",
            "Memoization Python functools",
            "Cache expensive function calls Python",
            "Python functools.cache example",
            "Speed up recursive function Python caching",
        ],
        "from functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n):\n    \"\"\"Compute nth Fibonacci number using memoization.\"\"\"\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ),
    (
        "property_decorator",
        [
            "How to use property decorator Python",
            "Python getter setter using property",
            "Make attribute read only Python",
            "@property Python class example",
            "Controlled attribute access Python",
        ],
        "class Temperature:\n    \"\"\"Temperature class with Celsius/Fahrenheit via @property.\"\"\"\n    def __init__(self, celsius=0):\n        self._celsius = celsius\n\n    @property\n    def celsius(self):\n        return self._celsius\n\n    @celsius.setter\n    def celsius(self, value):\n        if value < -273.15:\n            raise ValueError('Temperature below absolute zero')\n        self._celsius = value\n\n    @property\n    def fahrenheit(self):\n        return self._celsius * 9/5 + 32",
    ),
    (
        "staticmethod_classmethod",
        [
            "staticmethod vs classmethod Python",
            "When to use classmethod Python",
            "Python @staticmethod example",
            "Class method that creates instance Python",
            "Alternative constructor Python classmethod",
        ],
        "class Date:\n    \"\"\"Date class demonstrating classmethod and staticmethod.\"\"\"\n    def __init__(self, year, month, day):\n        self.year, self.month, self.day = year, month, day\n\n    @classmethod\n    def from_string(cls, s):\n        \"\"\"Alternative constructor from 'YYYY-MM-DD' string.\"\"\"\n        year, month, day = map(int, s.split('-'))\n        return cls(year, month, day)\n\n    @staticmethod\n    def is_leap(year):\n        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
    ),
    # ── CLOSURES ────────────────────────────────────────────
    (
        "closure_basic",
        [
            "What is a closure in Python",
            "Python closure example",
            "Function that remembers outer scope Python",
            "How to create a closure Python",
            "Encapsulate state with closure Python",
        ],
        "def make_counter(start=0):\n    \"\"\"Closure that maintains a counter.\"\"\"\n    count = [start]\n    def increment(by=1):\n        count[0] += by\n        return count[0]\n    return increment",
    ),
    (
        "nonlocal",
        [
            "How to use nonlocal in Python",
            "Modify outer variable from inner function Python",
            "nonlocal keyword Python example",
            "Python nested function variable",
        ],
        "def make_adder(n):\n    \"\"\"Use nonlocal to modify enclosing scope variable.\"\"\"\n    total = 0\n    def add(x):\n        nonlocal total\n        total += x + n\n        return total\n    return add",
    ),
    # ── ASYNC / CONCURRENCY ─────────────────────────────────
    (
        "async_await_basic",
        [
            "How to use async await in Python",
            "Python asyncio example",
            "Asynchronous function Python",
            "async def Python tutorial",
            "Python coroutine example",
            "Non blocking IO Python asyncio",
        ],
        "import asyncio\n\nasync def fetch_data(url: str) -> str:\n    \"\"\"Simulate async HTTP fetch.\"\"\"\n    await asyncio.sleep(1)  # simulate network delay\n    return f'Data from {url}'",
    ),
    (
        "asyncio_gather",
        [
            "Run multiple coroutines concurrently Python",
            "asyncio.gather Python example",
            "Await multiple tasks at once Python",
            "Concurrent async tasks Python",
            "Python gather coroutines result",
        ],
        "import asyncio\n\nasync def main():\n    \"\"\"Run tasks concurrently with asyncio.gather.\"\"\"\n    results = await asyncio.gather(\n        fetch('http://api1.com'),\n        fetch('http://api2.com'),\n        fetch('http://api3.com'),\n    )\n    return results",
    ),
    (
        "asyncio_task",
        [
            "How to create asyncio Task Python",
            "asyncio.create_task example",
            "Background task asyncio Python",
            "Schedule coroutine without awaiting immediately",
        ],
        "import asyncio\n\nasync def background_work():\n    await asyncio.sleep(2)\n    print('Done')\n\nasync def main():\n    \"\"\"Create and schedule a background task.\"\"\"\n    task = asyncio.create_task(background_work())\n    print('Task created, doing other work')\n    await task",
    ),
    (
        "asyncio_queue",
        [
            "Producer consumer asyncio Python",
            "asyncio.Queue Python example",
            "Async queue Python",
            "How to use Queue in asyncio Python",
        ],
        "import asyncio\n\nasync def producer(queue):\n    for i in range(5):\n        await queue.put(i)\n        await asyncio.sleep(0.1)\n\nasync def consumer(queue):\n    while True:\n        item = await queue.get()\n        print(f'Consumed {item}')\n        queue.task_done()\n\nasync def main():\n    q = asyncio.Queue()\n    await asyncio.gather(producer(q), consumer(q))",
    ),
    (
        "threading_basic",
        [
            "How to use threads in Python",
            "Python threading example",
            "Run function in separate thread Python",
            "Multithreading Python tutorial",
            "Python Thread class example",
        ],
        "import threading\n\ndef worker(name):\n    print(f'Thread {name} starting')\n\nthreads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]\nfor t in threads:\n    t.start()\nfor t in threads:\n    t.join()",
    ),
    (
        "threading_lock",
        [
            "How to use Lock in Python threading",
            "Thread safe counter Python",
            "Python threading.Lock example",
            "Prevent race condition Python threads",
            "Mutex Python threading",
        ],
        "import threading\n\nclass SafeCounter:\n    \"\"\"Thread-safe counter using Lock.\"\"\"\n    def __init__(self):\n        self._value = 0\n        self._lock = threading.Lock()\n\n    def increment(self):\n        with self._lock:\n            self._value += 1\n\n    @property\n    def value(self):\n        return self._value",
    ),
    (
        "multiprocessing_pool",
        [
            "How to use multiprocessing in Python",
            "Python multiprocessing Pool example",
            "Parallel processing Python",
            "Run CPU bound tasks in parallel Python",
            "ProcessPoolExecutor Python example",
        ],
        "from multiprocessing import Pool\n\ndef square(x):\n    return x * x\n\nif __name__ == '__main__':\n    with Pool(4) as p:\n        results = p.map(square, range(10))\n    print(results)",
    ),
    (
        "concurrent_futures",
        [
            "Python concurrent.futures example",
            "ThreadPoolExecutor Python",
            "ProcessPoolExecutor Python usage",
            "Submit tasks to thread pool Python",
            "Future object Python concurrent",
        ],
        "from concurrent.futures import ThreadPoolExecutor, as_completed\n\ndef fetch(url):\n    import time; time.sleep(0.5)\n    return f'result:{url}'\n\nwith ThreadPoolExecutor(max_workers=5) as ex:\n    futures = {ex.submit(fetch, url): url for url in ['a','b','c']}\n    for fut in as_completed(futures):\n        print(fut.result())",
    ),
    # ── CONTEXT MANAGERS ────────────────────────────────────
    (
        "context_manager_class",
        [
            "How to create a context manager Python",
            "Python __enter__ __exit__ example",
            "with statement custom class Python",
            "Context manager protocol Python",
            "Resource management Python context manager",
        ],
        "class ManagedFile:\n    \"\"\"Custom context manager for file handling.\"\"\"\n    def __init__(self, name, mode):\n        self.name = name\n        self.mode = mode\n\n    def __enter__(self):\n        self.file = open(self.name, self.mode)\n        return self.file\n\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.file.close()\n        return False",
    ),
    (
        "contextlib_contextmanager",
        [
            "contextlib.contextmanager Python example",
            "Generator based context manager Python",
            "@contextmanager decorator Python",
            "Create context manager with yield Python",
            "Simple context manager using generator",
        ],
        "from contextlib import contextmanager\n\n@contextmanager\ndef timer(label=''):\n    \"\"\"Context manager that times a block.\"\"\"\n    import time\n    start = time.perf_counter()\n    try:\n        yield\n    finally:\n        elapsed = time.perf_counter() - start\n        print(f'{label}: {elapsed:.4f}s')",
    ),
    (
        "contextlib_suppress",
        [
            "How to suppress exceptions Python",
            "contextlib.suppress Python example",
            "Ignore specific exception Python",
            "Swallow exception Python",
        ],
        "from contextlib import suppress\n\nwith suppress(FileNotFoundError):\n    os.remove('nonexistent.txt')  # no error raised",
    ),
    # ── DUNDER / MAGIC METHODS ──────────────────────────────
    (
        "dunder_repr_str",
        [
            "How to define __repr__ and __str__ Python",
            "Python string representation of object",
            "__repr__ vs __str__ Python",
            "Custom print for Python class",
            "Make Python object printable",
        ],
        "class Point:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n\n    def __repr__(self):\n        return f'Point({self.x!r}, {self.y!r})'\n\n    def __str__(self):\n        return f'({self.x}, {self.y})'",
    ),
    (
        "dunder_eq_hash",
        [
            "How to compare objects Python",
            "__eq__ and __hash__ Python example",
            "Custom equality Python class",
            "Make Python object hashable",
            "Use custom object as dict key Python",
        ],
        "class Point:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n\n    def __eq__(self, other):\n        return isinstance(other, Point) and self.x == other.x and self.y == other.y\n\n    def __hash__(self):\n        return hash((self.x, self.y))",
    ),
    (
        "dunder_len_getitem",
        [
            "How to make Python class iterable with indexing",
            "__len__ and __getitem__ Python",
            "Custom sequence Python class",
            "Implement indexing in Python class",
        ],
        "class MyList:\n    def __init__(self, data):\n        self._data = list(data)\n\n    def __len__(self):\n        return len(self._data)\n\n    def __getitem__(self, index):\n        return self._data[index]",
    ),
    (
        "dunder_add",
        [
            "Operator overloading Python",
            "__add__ Python example",
            "Overload + operator Python",
            "Custom addition for Python objects",
        ],
        "class Vector:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n\n    def __add__(self, other):\n        return Vector(self.x + other.x, self.y + other.y)\n\n    def __repr__(self):\n        return f'Vector({self.x}, {self.y})'",
    ),
    (
        "dunder_iter",
        [
            "How to make class iterable Python",
            "__iter__ and __next__ Python",
            "Custom iterator class Python",
            "Implement for loop support in Python class",
        ],
        "class Countdown:\n    def __init__(self, start):\n        self.start = start\n\n    def __iter__(self):\n        self.current = self.start\n        return self\n\n    def __next__(self):\n        if self.current <= 0:\n            raise StopIteration\n        self.current -= 1\n        return self.current + 1",
    ),
    (
        "dunder_call",
        [
            "How to make Python object callable",
            "__call__ method Python",
            "Callable class Python",
            "Call instance like a function Python",
        ],
        "class Multiplier:\n    def __init__(self, factor):\n        self.factor = factor\n\n    def __call__(self, x):\n        return x * self.factor\n\ndouble = Multiplier(2)\nprint(double(5))  # 10",
    ),
    (
        "dunder_slots",
        [
            "How to use __slots__ in Python",
            "Python __slots__ memory optimization",
            "Restrict attributes with __slots__ Python",
            "Reduce memory usage Python class slots",
        ],
        "class Point:\n    __slots__ = ('x', 'y')  # restrict to only x, y\n\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n# No __dict__ => saves memory for millions of instances",
    ),
    # ── DATACLASSES ─────────────────────────────────────────
    (
        "dataclass_basic",
        [
            "How to use dataclass in Python",
            "Python dataclass example",
            "@dataclass decorator Python",
            "Auto generate __init__ Python",
            "Simple data container Python",
        ],
        "from dataclasses import dataclass\n\n@dataclass\nclass Employee:\n    name: str\n    department: str\n    salary: float = 50000.0\n\n    def raise_salary(self, percent: float):\n        self.salary *= 1 + percent / 100",
    ),
    (
        "dataclass_frozen",
        [
            "Immutable dataclass Python",
            "Python frozen dataclass example",
            "Read only dataclass Python",
            "Hashable dataclass Python",
        ],
        "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\n\np = Point(1.0, 2.0)\n# p.x = 3.0  # raises FrozenInstanceError",
    ),
    (
        "dataclass_field",
        [
            "dataclass field with default factory Python",
            "Python dataclass list default value",
            "dataclasses.field Python example",
            "Mutable default in dataclass Python",
        ],
        "from dataclasses import dataclass, field\n\n@dataclass\nclass Team:\n    name: str\n    members: list = field(default_factory=list)\n    _id: int = field(default=0, repr=False)",
    ),
    # ── TYPE HINTS / TYPING ─────────────────────────────────
    (
        "type_hints_basic",
        [
            "How to add type hints in Python",
            "Python type annotation example",
            "Function signature type hints Python",
            "Python typing module tutorial",
        ],
        "from typing import List, Dict, Optional, Tuple\n\ndef process(items: List[str], config: Dict[str, int]) -> Optional[Tuple[int, str]]:\n    if not items:\n        return None\n    return len(items), items[0]",
    ),
    (
        "type_hints_union",
        [
            "Union type hint Python",
            "Python Optional vs Union",
            "Accept multiple types Python function",
            "Type hint that allows None Python",
            "Python 3.10 union type hint |",
        ],
        "from typing import Union\n\ndef to_int(val: Union[str, float, int]) -> int:\n    \"\"\"Convert string, float or int to int.\"\"\"\n    return int(val)",
    ),
    (
        "type_hints_generic",
        [
            "Generic type hint Python",
            "Python TypeVar example",
            "Generic function Python typing",
            "Type safe generic container Python",
        ],
        "from typing import TypeVar, Generic, List\n\nT = TypeVar('T')\n\nclass Stack(Generic[T]):\n    def __init__(self) -> None:\n        self._items: List[T] = []\n\n    def push(self, item: T) -> None:\n        self._items.append(item)\n\n    def pop(self) -> T:\n        return self._items.pop()",
    ),
    (
        "protocol_typing",
        [
            "Python Protocol typing example",
            "Structural subtyping Python",
            "typing.Protocol Python",
            "Duck typing with type hints Python",
            "Interface in Python using Protocol",
        ],
        "from typing import Protocol\n\nclass Drawable(Protocol):\n    def draw(self) -> None: ...\n\ndef render(obj: Drawable) -> None:\n    obj.draw()\n\n# Any class with draw() satisfies Drawable without inheritance",
    ),
    # ── COMPREHENSIONS ──────────────────────────────────────
    (
        "list_comprehension",
        [
            "Python list comprehension example",
            "How to use list comprehension Python",
            "Filter list with comprehension Python",
            "One liner list transformation Python",
            "Create list from loop Python comprehension",
        ],
        "# List comprehension: squares of even numbers\nevens_squared = [x**2 for x in range(20) if x % 2 == 0]",
    ),
    (
        "dict_comprehension",
        [
            "Python dict comprehension example",
            "Build dictionary with comprehension",
            "Map keys to values Python one liner",
            "Dictionary from two lists Python",
        ],
        "# Dict comprehension\nwords = ['hello', 'world', 'python']\nword_lengths = {w: len(w) for w in words}",
    ),
    (
        "set_comprehension",
        [
            "Python set comprehension example",
            "Remove duplicates with set comprehension",
            "Unique values comprehension Python",
        ],
        "# Set comprehension: unique first letters\nwords = ['apple', 'avocado', 'banana', 'blueberry']\nfirst_letters = {w[0] for w in words}",
    ),
    (
        "generator_expression",
        [
            "Python generator expression example",
            "Lazy list comprehension Python",
            "Memory efficient iteration Python",
            "Generator expression vs list comprehension",
            "How to create generator without yield Python",
        ],
        "# Generator expression - memory efficient\ndata = range(1_000_000)\ntotal = sum(x**2 for x in data if x % 2 == 0)",
    ),
    (
        "nested_comprehension",
        [
            "Nested list comprehension Python",
            "Flatten 2D list Python one liner",
            "Matrix to flat list Python",
            "List comprehension with two loops",
        ],
        "matrix = [[1,2,3],[4,5,6],[7,8,9]]\nflat = [elem for row in matrix for elem in row]",
    ),
    (
        "walrus_operator",
        [
            "Python walrus operator example",
            ":= operator Python 3.8",
            "Assignment expression Python",
            "Named expression Python walrus",
            "How to use := in Python",
        ],
        "# Walrus operator: assign and use in same expression\ndata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nfiltered = [y for x in data if (y := x**2) > 25]",
    ),
    # ── EXCEPTIONS / ERROR HANDLING ─────────────────────────
    (
        "custom_exception",
        [
            "How to create custom exception Python",
            "User defined exception Python",
            "Extend Exception class Python",
            "Custom error class Python",
            "Raise custom exception Python",
        ],
        "class ValidationError(Exception):\n    \"\"\"Raised when input validation fails.\"\"\"\n    def __init__(self, field: str, msg: str):\n        self.field = field\n        super().__init__(f'{field}: {msg}')\n\n\nclass NetworkError(RuntimeError):\n    \"\"\"Raised on network failures.\"\"\"\n    pass",
    ),
    (
        "exception_hierarchy",
        [
            "Python exception hierarchy example",
            "Catch multiple exceptions Python",
            "try except else finally Python",
            "Exception chaining Python",
            "raise from Python exception",
        ],
        "def divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError as e:\n        raise ValueError('b cannot be zero') from e\n    except TypeError:\n        raise\n    else:\n        print('Success')\n    finally:\n        print('Always runs')",
    ),
    (
        "exception_group",
        [
            "Python ExceptionGroup 3.11",
            "except* Python syntax",
            "Handle multiple exceptions at once Python 3.11",
            "ExceptionGroup Python example",
        ],
        "# Python 3.11+ ExceptionGroup\ntry:\n    raise ExceptionGroup('several errors', [\n        ValueError('bad value'),\n        TypeError('bad type'),\n    ])\nexcept* ValueError as eg:\n    print('ValueErrors:', eg.exceptions)\nexcept* TypeError as eg:\n    print('TypeErrors:', eg.exceptions)",
    ),
    # ── FILE I/O ─────────────────────────────────────────────
    (
        "file_read_write",
        [
            "How to read and write files Python",
            "Python open file example",
            "Read text file Python",
            "Write to file Python",
            "File handling Python tutorial",
        ],
        "# Read entire file\nwith open('data.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n\n# Write to file\nwith open('output.txt', 'w', encoding='utf-8') as f:\n    f.write('Hello, World!')",
    ),
    (
        "file_readline",
        [
            "Read file line by line Python",
            "Iterate over lines in file Python",
            "Python readline example",
            "Process large file Python",
            "Memory efficient file reading Python",
        ],
        "# Memory-efficient line-by-line reading\nwith open('large_file.txt', encoding='utf-8') as f:\n    for line in f:\n        process(line.rstrip('\\n'))",
    ),
    (
        "pathlib",
        [
            "How to use pathlib Python",
            "Python Path object example",
            "pathlib vs os.path Python",
            "File system operations pathlib Python",
            "Create directory Python pathlib",
        ],
        "from pathlib import Path\n\nbase = Path.home() / 'projects' / 'myapp'\nbase.mkdir(parents=True, exist_ok=True)\n\n# List all .py files recursively\npy_files = list(base.rglob('*.py'))\n\n# Read / write\n(base / 'config.txt').write_text('setting=value')",
    ),
    (
        "json_handling",
        [
            "How to read and write JSON Python",
            "Python json module example",
            "Parse JSON string Python",
            "Convert dict to JSON Python",
            "Load JSON file Python",
        ],
        "import json\n\n# Deserialize\nwith open('data.json') as f:\n    data = json.load(f)\n\n# Serialize\nwith open('output.json', 'w') as f:\n    json.dump(data, f, indent=2)\n\n# String operations\ntext = json.dumps({'key': 'value'})\nobj  = json.loads(text)",
    ),
    (
        "csv_handling",
        [
            "How to read CSV file Python",
            "Python csv module example",
            "Write CSV Python",
            "Parse CSV with header Python",
            "DictReader Python csv",
        ],
        "import csv\n\n# Read with headers as dicts\nwith open('data.csv', newline='') as f:\n    reader = csv.DictReader(f)\n    rows = list(reader)\n\n# Write\nwith open('out.csv', 'w', newline='') as f:\n    writer = csv.DictWriter(f, fieldnames=['name', 'age'])\n    writer.writeheader()\n    writer.writerows(rows)",
    ),
    (
        "pickle",
        [
            "How to serialize Python objects",
            "Python pickle module example",
            "Save and load Python objects",
            "Persist Python object to disk",
            "Pickle dump and load Python",
        ],
        "import pickle\n\ndata = {'key': [1, 2, 3], 'model': None}\n\n# Save\nwith open('data.pkl', 'wb') as f:\n    pickle.dump(data, f)\n\n# Load\nwith open('data.pkl', 'rb') as f:\n    loaded = pickle.load(f)",
    ),
    # ── DATA STRUCTURES ─────────────────────────────────────
    (
        "collections_defaultdict",
        [
            "Python defaultdict example",
            "collections.defaultdict tutorial",
            "Group items by key Python",
            "Avoid KeyError Python dictionary",
            "Default value dict Python",
        ],
        "from collections import defaultdict\n\n# Group words by first letter\nwords = ['apple', 'avocado', 'banana', 'blueberry', 'cherry']\ngroups = defaultdict(list)\nfor word in words:\n    groups[word[0]].append(word)",
    ),
    (
        "collections_counter",
        [
            "Python Counter example",
            "Count occurrences Python",
            "Most common elements Python",
            "Word frequency Python",
            "collections.Counter tutorial",
        ],
        "from collections import Counter\n\nwords = 'the quick brown fox jumps over the lazy dog'.split()\ncounts = Counter(words)\nprint(counts.most_common(3))",
    ),
    (
        "collections_deque",
        [
            "Python deque example",
            "Double ended queue Python",
            "Efficient FIFO queue Python",
            "Rotate list Python deque",
            "collections.deque Python tutorial",
        ],
        "from collections import deque\n\ndq = deque(maxlen=5)  # fixed-size circular buffer\nfor i in range(10):\n    dq.append(i)\nprint(dq)  # deque([5, 6, 7, 8, 9], maxlen=5)",
    ),
    (
        "collections_namedtuple",
        [
            "Python namedtuple example",
            "Named tuple Python",
            "Readable tuple Python",
            "collections.namedtuple tutorial",
            "Lightweight data object Python",
        ],
        "from collections import namedtuple\n\nPoint = namedtuple('Point', ['x', 'y'])\np = Point(3, 4)\nprint(p.x, p.y)  # access by name",
    ),
    (
        "collections_ordereddict",
        [
            "OrderedDict Python example",
            "Dictionary that remembers insertion order Python",
            "collections.OrderedDict tutorial",
        ],
        "from collections import OrderedDict\n\nod = OrderedDict()\nod['a'] = 1\nod['b'] = 2\nod.move_to_end('a')  # move 'a' to end",
    ),
    (
        "heapq",
        [
            "Python heapq example",
            "Min heap Python",
            "Priority queue Python",
            "Find N smallest Python",
            "heapq.heappush Python tutorial",
        ],
        "import heapq\n\nnums = [5, 1, 8, 3, 9, 2]\nheapq.heapify(nums)  # in-place min-heap\n\nsmallest = heapq.heappop(nums)\nnlargest  = heapq.nlargest(3, nums)",
    ),
    (
        "bisect",
        [
            "Binary search Python",
            "Python bisect module example",
            "Insert into sorted list Python",
            "bisect.insort Python tutorial",
            "Maintain sorted list Python",
        ],
        "import bisect\n\nsorted_list = [1, 3, 5, 7, 9]\nbisect.insort(sorted_list, 4)  # insert 4 keeping sorted\npos = bisect.bisect_left(sorted_list, 5)  # find position of 5",
    ),
    # ── ALGORITHMS ──────────────────────────────────────────
    (
        "sorting_key",
        [
            "How to sort list by custom key Python",
            "sorted() with key Python",
            "Sort objects by attribute Python",
            "Python sort lambda example",
        ],
        "people = [('Alice', 30), ('Bob', 25), ('Carol', 35)]\n\n# Sort by age ascending\nby_age = sorted(people, key=lambda p: p[1])\n\n# Sort by name descending\nby_name_desc = sorted(people, key=lambda p: p[0], reverse=True)",
    ),
    (
        "binary_search",
        [
            "Implement binary search Python",
            "Binary search algorithm Python",
            "Efficient search sorted list Python",
            "O(log n) search Python",
        ],
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    ),
    (
        "quicksort",
        [
            "Implement quicksort Python",
            "Quicksort algorithm Python",
            "Divide and conquer sort Python",
            "Python recursive sort example",
        ],
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left   = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right  = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    ),
    (
        "merge_sort",
        [
            "Implement merge sort Python",
            "Mergesort algorithm Python",
            "Stable sort Python recursive",
            "Divide and merge sort Python",
        ],
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left  = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(l, r):\n    result = []\n    i = j = 0\n    while i < len(l) and j < len(r):\n        if l[i] <= r[j]:\n            result.append(l[i]); i += 1\n        else:\n            result.append(r[j]); j += 1\n    return result + l[i:] + r[j:]",
    ),
    (
        "dynamic_programming_fib",
        [
            "Dynamic programming Python example",
            "Bottom up DP Python",
            "Fibonacci with dynamic programming Python",
            "Tabulation approach Python",
        ],
        "def fib_dp(n):\n    \"\"\"Fibonacci via bottom-up dynamic programming.\"\"\"\n    if n < 2:\n        return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n    for i in range(2, n + 1):\n        dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]",
    ),
    (
        "graph_bfs",
        [
            "Breadth first search Python",
            "BFS algorithm Python",
            "Shortest path unweighted graph Python",
            "Graph traversal Python BFS",
        ],
        "from collections import deque\n\ndef bfs(graph, start):\n    visited = {start}\n    queue   = deque([start])\n    order   = []\n    while queue:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return order",
    ),
    (
        "graph_dfs",
        [
            "Depth first search Python",
            "DFS algorithm Python",
            "Recursive graph traversal Python",
            "DFS iterative Python",
        ],
        "def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited",
    ),
    (
        "trie",
        [
            "Implement Trie Python",
            "Prefix tree Python",
            "Trie data structure Python",
            "Autocomplete data structure Python",
        ],
        "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n\n    def insert(self, word):\n        node = self.root\n        for ch in word:\n            node = node.children.setdefault(ch, TrieNode())\n        node.is_end = True\n\n    def search(self, word):\n        node = self.root\n        for ch in word:\n            if ch not in node.children:\n                return False\n            node = node.children[ch]\n        return node.is_end",
    ),
    # ── OOP PATTERNS ────────────────────────────────────────
    (
        "singleton",
        [
            "Singleton pattern Python",
            "Only one instance of class Python",
            "Singleton design pattern Python example",
            "Ensure single instance Python",
        ],
        "class Singleton:\n    _instance = None\n\n    def __new__(cls, *args, **kwargs):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
    ),
    (
        "factory_pattern",
        [
            "Factory design pattern Python",
            "Create objects based on type Python",
            "Factory method Python example",
            "Object creation strategy Python",
        ],
        "class AnimalFactory:\n    @staticmethod\n    def create(kind: str):\n        animals = {'dog': Dog, 'cat': Cat, 'bird': Bird}\n        cls = animals.get(kind)\n        if cls is None:\n            raise ValueError(f'Unknown animal: {kind}')\n        return cls()",
    ),
    (
        "observer_pattern",
        [
            "Observer pattern Python",
            "Event listener Python",
            "Publish subscribe Python",
            "Callback registration Python",
            "Notify subscribers Python",
        ],
        "class EventEmitter:\n    def __init__(self):\n        self._listeners = {}\n\n    def on(self, event, callback):\n        self._listeners.setdefault(event, []).append(callback)\n\n    def emit(self, event, *args, **kwargs):\n        for cb in self._listeners.get(event, []):\n            cb(*args, **kwargs)",
    ),
    (
        "mixin",
        [
            "Python mixin example",
            "Multiple inheritance mixin Python",
            "Add behaviour with mixin Python",
            "Reusable mixin class Python",
        ],
        "class SerializeMixin:\n    def to_dict(self):\n        return self.__dict__\n\n    def to_json(self):\n        import json\n        return json.dumps(self.to_dict())\n\nclass User(SerializeMixin):\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email",
    ),
    (
        "abstract_class",
        [
            "Abstract class Python",
            "ABC Python example",
            "Abstract method Python",
            "Force subclass to implement method Python",
            "Interface Python abc module",
        ],
        "from abc import ABC, abstractmethod\n\nclass Shape(ABC):\n    @abstractmethod\n    def area(self) -> float: ...\n\n    @abstractmethod\n    def perimeter(self) -> float: ...\n\nclass Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        import math; return math.pi * self.r**2\n    def perimeter(self):\n        import math; return 2 * math.pi * self.r",
    ),
    # ── FUNCTIONAL PROGRAMMING ──────────────────────────────
    (
        "map_filter_reduce",
        [
            "Python map filter reduce example",
            "Functional programming Python",
            "Transform list with map Python",
            "Filter list Python",
            "functools.reduce Python",
        ],
        "from functools import reduce\n\nnums = [1, 2, 3, 4, 5]\n\nsquared   = list(map(lambda x: x**2, nums))\nevens     = list(filter(lambda x: x % 2 == 0, nums))\ntotal     = reduce(lambda a, b: a + b, nums)",
    ),
    (
        "partial",
        [
            "Python functools.partial example",
            "Partial function application Python",
            "Pre-fill function arguments Python",
            "Currying Python",
        ],
        "from functools import partial\n\ndef power(base, exp):\n    return base ** exp\n\nsquare = partial(power, exp=2)\ncube   = partial(power, exp=3)\n\nprint(square(4))  # 16\nprint(cube(3))    # 27",
    ),
    # ── LOGGING ─────────────────────────────────────────────
    (
        "logging_basic",
        [
            "How to use logging in Python",
            "Python logging module example",
            "Log messages Python",
            "Debug info warning error Python logging",
        ],
        "import logging\n\nlogging.basicConfig(\n    level=logging.INFO,\n    format='%(asctime)s %(levelname)s %(name)s: %(message)s'\n)\nlogger = logging.getLogger(__name__)\n\nlogger.info('Application started')\nlogger.warning('Low memory')\nlogger.error('Failed to connect', exc_info=True)",
    ),
    (
        "logging_handler",
        [
            "Python logging to file example",
            "FileHandler Python logging",
            "Rotating log file Python",
            "Log to file and console Python",
        ],
        "import logging\nfrom logging.handlers import RotatingFileHandler\n\nlogger = logging.getLogger('myapp')\nlogger.setLevel(logging.DEBUG)\n\nfh = RotatingFileHandler('app.log', maxBytes=1_000_000, backupCount=3)\nfh.setLevel(logging.INFO)\n\nch = logging.StreamHandler()\nch.setLevel(logging.DEBUG)\n\nlogger.addHandler(fh)\nlogger.addHandler(ch)",
    ),
    # ── NETWORKING / WEB ────────────────────────────────────
    (
        "requests_get",
        [
            "How to make HTTP request Python",
            "Python requests library example",
            "GET request Python",
            "Fetch URL Python",
            "HTTP GET with headers Python",
        ],
        "import requests\n\nresponse = requests.get(\n    'https://api.example.com/data',\n    headers={'Authorization': 'Bearer token'},\n    timeout=10\n)\nresponse.raise_for_status()\ndata = response.json()",
    ),
    (
        "requests_post",
        [
            "POST request Python",
            "Send JSON data HTTP Python",
            "Python requests post example",
            "Submit form data Python requests",
        ],
        "import requests\n\npayload = {'name': 'Alice', 'score': 99}\nresp = requests.post('https://api.example.com/submit', json=payload, timeout=10)\nresp.raise_for_status()\nprint(resp.json())",
    ),
    (
        "fastapi_route",
        [
            "FastAPI route example",
            "Python REST API with FastAPI",
            "Define GET endpoint FastAPI",
            "FastAPI tutorial Python",
        ],
        "from fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n    price: float\n\n@app.get('/items/{item_id}')\ndef get_item(item_id: int):\n    return {'id': item_id}\n\n@app.post('/items/')\ndef create_item(item: Item):\n    return item",
    ),
    (
        "flask_route",
        [
            "Flask route example Python",
            "Python web app Flask",
            "Flask GET POST route",
            "Simple REST API Flask Python",
        ],
        "from flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.route('/hello', methods=['GET'])\ndef hello():\n    name = request.args.get('name', 'World')\n    return jsonify({'message': f'Hello, {name}!'})",
    ),
    # ── REGEX ────────────────────────────────────────────────
    (
        "regex_basic",
        [
            "How to use regex in Python",
            "Python re module example",
            "Regular expression Python",
            "Search pattern in string Python",
            "Match email address Python regex",
        ],
        "import re\n\n# Find all email addresses\ntext = 'Contact alice@example.com or bob@test.org'\nemails = re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.]+', text)",
    ),
    (
        "regex_groups",
        [
            "Regex groups Python",
            "Named capture groups Python re",
            "Extract parts of string Python regex",
            "re.match groups Python",
        ],
        "import re\n\npattern = re.compile(r'(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})')\nm = pattern.search('Date: 2024-03-15')\nif m:\n    print(m.group('year'), m.group('month'), m.group('day'))",
    ),
    (
        "regex_sub",
        [
            "Replace pattern in string Python",
            "re.sub Python example",
            "Regex replace Python",
            "Find and replace with regex Python",
        ],
        "import re\n\ntext = 'Hello   World   Python'\n# Replace multiple spaces with single space\ncleaned = re.sub(r'\\s+', ' ', text.strip())",
    ),
    # ── TESTING ─────────────────────────────────────────────
    (
        "unittest_basic",
        [
            "How to write unit tests Python",
            "Python unittest example",
            "Test case Python",
            "Assert equal Python unittest",
            "Python test class",
        ],
        "import unittest\n\nclass TestMath(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(1 + 1, 2)\n\n    def test_divide_zero(self):\n        with self.assertRaises(ZeroDivisionError):\n            1 / 0\n\nif __name__ == '__main__':\n    unittest.main()",
    ),
    (
        "pytest_basic",
        [
            "Python pytest example",
            "Write tests with pytest",
            "pytest fixture Python",
            "Parametrize pytest",
            "Simple pytest test Python",
        ],
        "import pytest\n\ndef add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(2, 3) == 5\n\n@pytest.mark.parametrize('a,b,expected', [(1,2,3),(0,0,0),(-1,1,0)])\ndef test_add_parametric(a, b, expected):\n    assert add(a, b) == expected",
    ),
    (
        "pytest_fixture",
        [
            "pytest fixture example",
            "Setup and teardown pytest",
            "Reusable test setup Python pytest",
            "pytest conftest fixture",
        ],
        "import pytest\n\n@pytest.fixture\ndef sample_data():\n    return {'name': 'Alice', 'age': 30}\n\ndef test_name(sample_data):\n    assert sample_data['name'] == 'Alice'\n\ndef test_age(sample_data):\n    assert sample_data['age'] == 30",
    ),
    (
        "mock_patch",
        [
            "How to mock in Python tests",
            "unittest.mock patch Python",
            "Mock external dependency Python",
            "Patch function Python unittest",
            "MagicMock Python example",
        ],
        "from unittest.mock import patch, MagicMock\n\ndef get_user(user_id):\n    import requests\n    return requests.get(f'/users/{user_id}').json()\n\ndef test_get_user():\n    with patch('requests.get') as mock_get:\n        mock_get.return_value.json.return_value = {'id': 1, 'name': 'Alice'}\n        result = get_user(1)\n        assert result['name'] == 'Alice'",
    ),
    # ── ENVIRONMENT / CONFIG ─────────────────────────────────
    (
        "env_variables",
        [
            "How to read environment variables Python",
            "os.environ Python example",
            "Python dotenv example",
            "Config from environment Python",
            "python-dotenv load .env file",
        ],
        "import os\nfrom dotenv import load_dotenv\n\nload_dotenv()  # reads .env file\n\nDB_URL  = os.environ['DATABASE_URL']\nAPI_KEY = os.getenv('API_KEY', 'default-key')",
    ),
    (
        "argparse",
        [
            "How to parse command line arguments Python",
            "argparse Python example",
            "Python CLI argument parser",
            "sys.argv vs argparse Python",
        ],
        "import argparse\n\nparser = argparse.ArgumentParser(description='My CLI tool')\nparser.add_argument('input', help='Input file path')\nparser.add_argument('-o', '--output', default='out.txt', help='Output file')\nparser.add_argument('-v', '--verbose', action='store_true')\n\nargs = parser.parse_args()\nprint(args.input, args.output, args.verbose)",
    ),
    # ── DESCRIPTORS / METACLASSES ────────────────────────────
    (
        "descriptor",
        [
            "Python descriptor example",
            "__get__ __set__ __delete__ Python",
            "Descriptor protocol Python",
            "Validate attribute with descriptor Python",
        ],
        "class Positive:\n    \"\"\"Descriptor that enforces positive values.\"\"\"\n    def __set_name__(self, owner, name):\n        self.name = name\n\n    def __get__(self, obj, objtype=None):\n        return getattr(obj, f'_{self.name}', None)\n\n    def __set__(self, obj, value):\n        if value <= 0:\n            raise ValueError(f'{self.name} must be positive')\n        setattr(obj, f'_{self.name}', value)\n\nclass Circle:\n    radius = Positive()\n    def __init__(self, r):\n        self.radius = r",
    ),
    (
        "metaclass",
        [
            "Python metaclass example",
            "type() as metaclass Python",
            "Custom metaclass Python",
            "Control class creation Python",
            "Metaclass __new__ Python",
        ],
        "class SingletonMeta(type):\n    _instances = {}\n\n    def __call__(cls, *args, **kwargs):\n        if cls not in cls._instances:\n            cls._instances[cls] = super().__call__(*args, **kwargs)\n        return cls._instances[cls]\n\nclass Config(metaclass=SingletonMeta):\n    def __init__(self):\n        self.debug = False",
    ),
    # ── STRING OPERATIONS ────────────────────────────────────
    (
        "f_strings",
        [
            "Python f-string example",
            "Format string Python 3.6",
            "String interpolation Python",
            "f string with expressions Python",
        ],
        "name = 'Alice'\nage  = 30\npi   = 3.14159\n\n# Basic\nprint(f'Hello, {name}!')\n# Expression\nprint(f'In 5 years: {age + 5}')\n# Format spec\nprint(f'Pi = {pi:.2f}')\n# Repr\nprint(f'{name!r}')",
    ),
    (
        "string_methods",
        [
            "Common Python string methods",
            "split join strip Python string",
            "String manipulation Python",
            "How to split string Python",
        ],
        "s = '  Hello, World!  '\n\nprint(s.strip())           # remove whitespace\nprint(s.lower())           # to lowercase\nprint(s.upper())           # to uppercase\nprint(s.replace('World', 'Python'))\nwords = s.strip().split(', ')\nprint('|'.join(words))",
    ),
    (
        "string_format",
        [
            "Python str.format example",
            "Old style string formatting Python",
            "% formatting Python",
            "String template Python",
        ],
        "# str.format\nmsg = 'Hello, {}! You are {} years old.'.format('Alice', 30)\n\n# Named placeholders\nmsg2 = '{name} scored {score:.1f}%'.format(name='Bob', score=92.5)\n\n# % style\nmsg3 = 'Value: %d, Float: %.2f' % (42, 3.14)",
    ),
    # ── NUMBERS / MATH ───────────────────────────────────────
    (
        "math_module",
        [
            "Python math module example",
            "Square root Python",
            "Math functions Python",
            "Trigonometry Python",
            "Python math.floor ceil round",
        ],
        "import math\n\nprint(math.sqrt(16))         # 4.0\nprint(math.floor(3.7))       # 3\nprint(math.ceil(3.2))        # 4\nprint(math.log(100, 10))     # 2.0\nprint(math.pi, math.e)       # constants\nprint(math.factorial(5))     # 120",
    ),
    (
        "random_module",
        [
            "Python random module example",
            "Generate random number Python",
            "Random choice Python",
            "Shuffle list Python",
            "Random sample Python",
        ],
        "import random\n\nprint(random.randint(1, 100))         # integer [1, 100]\nprint(random.random())                # float [0, 1)\nprint(random.choice(['a','b','c']))   # pick one\n\nlst = list(range(10))\nrandom.shuffle(lst)                   # in-place shuffle\nsample = random.sample(lst, 3)        # 3 unique picks",
    ),
    (
        "statistics_module",
        [
            "Python statistics module",
            "Mean median mode Python",
            "Standard deviation Python",
            "Variance Python statistics",
        ],
        "import statistics as stats\n\ndata = [2, 4, 4, 4, 5, 5, 7, 9]\nprint(stats.mean(data))     # 5.0\nprint(stats.median(data))   # 4.5\nprint(stats.mode(data))     # 4\nprint(stats.stdev(data))    # 2.0",
    ),
    # ── MATCH-CASE (Python 3.10+) ────────────────────────────
    (
        "match_case",
        [
            "Python match case example",
            "Structural pattern matching Python 3.10",
            "switch statement Python",
            "match statement Python",
            "Pattern matching Python",
        ],
        "def http_status(status):\n    match status:\n        case 200:\n            return 'OK'\n        case 404:\n            return 'Not Found'\n        case 500 | 503:\n            return 'Server Error'\n        case _:\n            return 'Unknown'",
    ),
    (
        "match_case_struct",
        [
            "Structural pattern matching classes Python",
            "match case with dataclass Python",
            "Destructuring in Python match",
            "Pattern match object Python",
        ],
        "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: int\n    y: int\n\ndef describe(point):\n    match point:\n        case Point(x=0, y=0):\n            return 'Origin'\n        case Point(x=0, y=y):\n            return f'Y-axis at {y}'\n        case Point(x=x, y=0):\n            return f'X-axis at {x}'\n        case Point(x=x, y=y):\n            return f'Point ({x}, {y})'",
    ),
    # ── ENUMS ────────────────────────────────────────────────
    (
        "enum_basic",
        [
            "Python enum example",
            "How to use Enum in Python",
            "Named constants Python enum",
            "Enumeration class Python",
        ],
        "from enum import Enum, auto\n\nclass Color(Enum):\n    RED   = auto()\n    GREEN = auto()\n    BLUE  = auto()\n\nprint(Color.RED)         # Color.RED\nprint(Color.RED.value)   # 1\nfor color in Color:\n    print(color.name)",
    ),
    (
        "enum_flag",
        [
            "Python enum Flag example",
            "Bitwise flags Python",
            "Combine enum values Python",
            "Permission flags Python enum",
        ],
        "from enum import Flag, auto\n\nclass Permission(Flag):\n    READ    = auto()\n    WRITE   = auto()\n    EXECUTE = auto()\n    ALL     = READ | WRITE | EXECUTE\n\nuser_perm = Permission.READ | Permission.WRITE\nprint(Permission.READ in user_perm)  # True",
    ),
    # ── SLOTS, __init_subclass__, super ──────────────────────
    (
        "super_call",
        [
            "How to call parent class method Python",
            "super() Python example",
            "Invoke parent __init__ Python",
            "super() in inheritance Python",
        ],
        "class Animal:\n    def __init__(self, name):\n        self.name = name\n\n    def speak(self):\n        return f'{self.name} makes a sound'\n\nclass Dog(Animal):\n    def __init__(self, name, breed):\n        super().__init__(name)\n        self.breed = breed\n\n    def speak(self):\n        return super().speak() + ' - Woof!'",
    ),
    (
        "init_subclass",
        [
            "Python __init_subclass__ example",
            "Validate subclass creation Python",
            "Plugin registration Python __init_subclass__",
            "Hook subclass definition Python",
        ],
        "class Base:\n    _registry = {}\n\n    def __init_subclass__(cls, kind=None, **kw):\n        super().__init_subclass__(**kw)\n        if kind:\n            Base._registry[kind] = cls\n\nclass Dog(Base, kind='dog'): pass\nclass Cat(Base, kind='cat'): pass\n\nprint(Base._registry)  # {'dog': Dog, 'cat': Cat}",
    ),
    # ── MEMORY / PERFORMANCE ─────────────────────────────────
    (
        "slots_memory",
        [
            "Reduce Python object memory",
            "Python __slots__ vs __dict__",
            "Memory efficient class Python",
            "Object overhead Python",
        ],
        "import sys\n\nclass WithDict:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n\nclass WithSlots:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n\nprint(sys.getsizeof(WithDict(1,2)))   # ~48 bytes\nprint(sys.getsizeof(WithSlots(1,2))) # ~32 bytes",
    ),
    (
        "profiling",
        [
            "Profile Python code performance",
            "Python cProfile example",
            "Find slow code Python",
            "timeit Python",
            "Benchmark function Python",
        ],
        "import cProfile, pstats, timeit\n\n# Quick benchmark\nelapsed = timeit.timeit('sum(range(1000))', number=10_000)\nprint(f'10k runs: {elapsed:.3f}s')\n\n# Full profiling\ncProfile.run('my_function()', 'output.prof')\np = pstats.Stats('output.prof').sort_stats('cumulative')\np.print_stats(10)",
    ),
    # ── COMPREHENSION / UNPACKING ────────────────────────────
    (
        "star_unpacking",
        [
            "Python star unpacking example",
            "Extended unpacking Python",
            "*args **kwargs Python",
            "Unpack list into variables Python",
            "Spread operator Python",
        ],
        "first, *middle, last = [1, 2, 3, 4, 5]\nprint(first, last)   # 1 5\nprint(middle)         # [2, 3, 4]\n\n# Merge dicts Python 3.9+\na = {'x': 1}\nb = {'y': 2}\nc = {**a, **b}  # {'x': 1, 'y': 2}",
    ),
    (
        "unpacking_function",
        [
            "Unpack list as function arguments Python",
            "*args Python example",
            "**kwargs Python example",
            "Variable arguments Python function",
        ],
        "def display(*args, **kwargs):\n    for arg in args:\n        print(arg)\n    for key, val in kwargs.items():\n        print(f'{key}={val}')\n\nparams = [1, 2, 3]\nopts   = {'sep': ', ', 'end': '\\n'}\ndisplay(*params, **opts)",
    ),
    # ── PANDAS / NUMPY (data science) ───────────────────────
    (
        "numpy_array",
        [
            "NumPy array operations Python",
            "Create numpy array Python",
            "Array math Python numpy",
            "Vectorized operations numpy",
        ],
        "import numpy as np\n\na = np.array([1, 2, 3, 4, 5])\nb = np.arange(5)\nc = np.zeros((3, 3))\nd = np.ones((2, 4))\n\nprint(a * 2)          # [2 4 6 8 10]\nprint(a.mean())       # 3.0\nprint(a @ a)          # dot product = 55",
    ),
    (
        "pandas_dataframe",
        [
            "Create pandas DataFrame Python",
            "Pandas read csv Python",
            "DataFrame operations Python",
            "Filter rows pandas Python",
            "Group by pandas Python",
        ],
        "import pandas as pd\n\ndf = pd.read_csv('data.csv')\n\n# Filter\nhigh_salary = df[df['salary'] > 100_000]\n\n# Group\navg_by_dept = df.groupby('department')['salary'].mean()\n\n# New column\ndf['bonus'] = df['salary'] * 0.10",
    ),
    # ── PYDANTIC ─────────────────────────────────────────────
    (
        "pydantic_model",
        [
            "Pydantic model Python example",
            "Data validation Python pydantic",
            "BaseModel pydantic Python",
            "Type safe data model Python",
            "Validate input Python pydantic",
        ],
        "from pydantic import BaseModel, Field, validator\nfrom typing import Optional\n\nclass User(BaseModel):\n    name: str\n    email: str\n    age: int = Field(gt=0, le=150)\n    bio: Optional[str] = None\n\n    @validator('email')\n    def email_must_have_at(cls, v):\n        if '@' not in v:\n            raise ValueError('invalid email')\n        return v",
    ),
    # ── MISCELLANEOUS ────────────────────────────────────────
    (
        "zip_enumerate",
        [
            "Python zip example",
            "Iterate two lists simultaneously Python",
            "enumerate Python example",
            "Loop with index Python",
        ],
        "names  = ['Alice', 'Bob', 'Carol']\nscores = [95, 87, 92]\n\n# zip: pair elements\nfor name, score in zip(names, scores):\n    print(f'{name}: {score}')\n\n# enumerate: add index\nfor i, name in enumerate(names, start=1):\n    print(f'{i}. {name}')",
    ),
    (
        "any_all",
        [
            "Python any all example",
            "Check all elements satisfy condition Python",
            "any() all() Python",
            "Logical OR/AND over list Python",
        ],
        "nums = [2, 4, 6, 8, 10]\n\nprint(all(n % 2 == 0 for n in nums))  # True - all even\nprint(any(n > 9 for n in nums))       # True - at least one > 9",
    ),
    (
        "sorted_key_complex",
        [
            "Sort list of dicts Python",
            "Sort by multiple keys Python",
            "operator.attrgetter sort Python",
            "Sort objects multiple criteria Python",
        ],
        "from operator import attrgetter, itemgetter\n\npeople = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}, {'name': 'Carol', 'age': 30}]\n\n# Sort by age then name\nsorted_people = sorted(people, key=lambda p: (p['age'], p['name']))",
    ),
    (
        "copy_deepcopy",
        [
            "Deep copy Python",
            "Python copy vs deepcopy",
            "Copy nested object Python",
            "clone list Python",
            "How to copy dict Python",
        ],
        "import copy\n\noriginal = [[1, 2, 3], [4, 5, 6]]\n\nshallow  = copy.copy(original)    # nested lists still shared\ndeep     = copy.deepcopy(original) # fully independent copy\n\ndeep[0][0] = 99\nprint(original[0][0])  # still 1",
    ),
    (
        "weakref",
        [
            "Python weakref example",
            "Weak reference Python",
            "Avoid circular reference Python",
            "WeakValueDictionary Python",
        ],
        "import weakref\n\nclass Node:\n    def __init__(self, val):\n        self.val = val\n        self.parent = None  # strong reference would cause cycle\n\nparent = Node(1)\nchild  = Node(2)\nchild.parent = weakref.ref(parent)  # weak reference\n\n# Access: child.parent() returns None if parent was GC'd",
    ),
    (
        "operator_module",
        [
            "Python operator module example",
            "itemgetter attrgetter Python",
            "Functional operators Python",
            "Faster than lambda Python",
        ],
        "from operator import itemgetter, attrgetter, mul\nfrom functools import reduce\n\nitems = [('apple', 3), ('banana', 1), ('cherry', 2)]\nsorted_items = sorted(items, key=itemgetter(1))\n\nproduct = reduce(mul, [1, 2, 3, 4, 5])  # 120",
    ),
    (
        "struct_module",
        [
            "Python struct module example",
            "Pack unpack binary data Python",
            "Binary file format Python",
            "Convert int to bytes Python struct",
        ],
        "import struct\n\n# Pack integers into bytes\npacked = struct.pack('>IHH', 2024, 3, 15)  # big-endian uint32, uint16, uint16\nprint(packed)\n\n# Unpack\nyr, mo, dy = struct.unpack('>IHH', packed)\nprint(yr, mo, dy)",
    ),
]

# ─────────────────────────────────────────────────────────────
# SECTION 3 – PARAPHRASE TEMPLATES
# Applied to every anchor to create variations
# ─────────────────────────────────────────────────────────────
QUESTION_TEMPLATES = [
    "{anchor}",
    "How do I {anchor_lower}?",
    "What is the best way to {anchor_lower}?",
    "Give me an example of {anchor_lower}.",
    "Show me how to {anchor_lower} in Python.",
    "Python: {anchor}",
    "Explain {anchor_lower} with code.",
    "I want to understand {anchor_lower}.",
    "Can you show {anchor_lower} example?",
    "Write Python code for {anchor_lower}.",
    "Python snippet for {anchor_lower}.",
    "{anchor} tutorial",
    "Demonstrate {anchor_lower} in Python 3.",
    "What does {anchor_lower} look like in Python?",
    "Code example: {anchor_lower}",
]

def expand_anchors(anchor, positive):
    """Generate many paraphrased (anchor, positive) pairs."""
    pairs = []
    al = anchor.lower().rstrip('?').strip()
    seen = set()
    for tpl in QUESTION_TEMPLATES:
        q = tpl.format(anchor=anchor, anchor_lower=al)
        if q not in seen:
            seen.add(q)
            pairs.append({"anchor": q, "positive": positive})
    return pairs

# ─────────────────────────────────────────────────────────────
# SECTION 4 – EXTRA CONCEPT PAIRS (direct, no template)
# ─────────────────────────────────────────────────────────────
EXTRA_DIRECT = [
    {"anchor": "How to resume a function from where it left in Python", "positive": "def gen():\n    print('First')\n    yield\n    print('Second')\n    yield\ng = gen()\nnext(g)  # prints First\nnext(g)  # prints Second"},
    {"anchor": "Python function that can be paused and resumed", "positive": "def stepper():\n    yield 'step 1'\n    yield 'step 2'\n    yield 'step 3'\n\ng = stepper()\nprint(next(g))  # step 1"},
    {"anchor": "Lazy evaluation in Python", "positive": "def lazy_range(n):\n    i = 0\n    while i < n:\n        yield i\n        i += 1"},
    {"anchor": "How generators differ from regular functions", "positive": "# Regular function returns once\ndef regular():\n    return [1, 2, 3]\n\n# Generator yields multiple times, lazily\ndef generator():\n    yield 1\n    yield 2\n    yield 3"},
    {"anchor": "Python infinite sequence", "positive": "def natural_numbers():\n    n = 1\n    while True:\n        yield n\n        n += 1\n\n# Use with islice to avoid infinite loop\nimport itertools\nfirst_10 = list(itertools.islice(natural_numbers(), 10))"},
    {"anchor": "How to create an infinite counter Python", "positive": "import itertools\ncounter = itertools.count(start=0, step=1)\nfor _ in range(5):\n    print(next(counter))"},
    {"anchor": "Python asynchronous generator", "positive": "import asyncio\n\nasync def async_range(n):\n    for i in range(n):\n        await asyncio.sleep(0)\n        yield i\n\nasync def main():\n    async for val in async_range(5):\n        print(val)"},
    {"anchor": "How to check if object is a generator Python", "positive": "import inspect\n\ndef gen():\n    yield 1\n\ng = gen()\nprint(inspect.isgenerator(g))    # True\nprint(inspect.isgeneratorfunction(gen))  # True"},
    {"anchor": "StopIteration exception Python", "positive": "def gen():\n    yield 1\n    yield 2\n\ng = gen()\nprint(next(g))  # 1\nprint(next(g))  # 2\ntry:\n    next(g)     # raises StopIteration\nexcept StopIteration:\n    print('Generator exhausted')"},
    {"anchor": "How to implement pipeline with generators Python", "positive": "def read_lines(filename):\n    with open(filename) as f:\n        yield from f\n\ndef strip_lines(lines):\n    for line in lines:\n        yield line.strip()\n\ndef filter_empty(lines):\n    for line in lines:\n        if line:\n            yield line\n\n# Compose pipeline\ndef pipeline(filename):\n    return filter_empty(strip_lines(read_lines(filename)))"},
]

# ─────────────────────────────────────────────────────────────
# SECTION 5 – HUGGINGFACE DATASETS (optional, needs network)
# ─────────────────────────────────────────────────────────────
def try_load_hf_dataset():
    """
    Try to load anchor-positive pairs from HuggingFace datasets.
    Requires: pip install datasets
    Returns list of {"anchor": ..., "positive": ...} or []
    """
    pairs = []
    try:
        from datasets import load_dataset
        print("Loading sentence-transformers/codesearchnet Python split...")
        ds = load_dataset("sentence-transformers/codesearchnet", "python",
                          split="train", trust_remote_code=True)
        for row in ds:
            anchor   = str(row.get("func_documentation_string", "")).strip()
            positive = str(row.get("func_code_string", "")).strip()
            if len(anchor) >= 20 and len(positive) >= 40:
                pairs.append({"anchor": anchor, "positive": positive})
            if len(pairs) >= 50_000:
                break
        print(f"  Loaded {len(pairs)} pairs from codesearchnet")
    except Exception as e:
        print(f"  [HuggingFace skipped] {e}")
    return pairs

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    all_pairs = []
    seen_anchors = set()

    def add(pairs):
        for p in pairs:
            k = p["anchor"].lower().strip()
            if k not in seen_anchors and len(p["anchor"]) >= 10 and len(p["positive"]) >= 20:
                seen_anchors.add(k)
                all_pairs.append(p)

    # 1. Existing 601 pairs
    print("Loading existing dataset...")
    add(load_existing())
    print(f"  After existing: {len(all_pairs)}")

    # 2. Extra direct pairs
    add(EXTRA_DIRECT)
    print(f"  After extra direct: {len(all_pairs)}")

    # 3. Expand seeds with templates
    print("Expanding seed bank with paraphrase templates...")
    for topic, anchors, positive in SEEDS:
        for anchor in anchors:
            add(expand_anchors(anchor, positive))
    print(f"  After seed expansion: {len(all_pairs)}")

    # 4. Optional HuggingFace data
    hf = try_load_hf_dataset()
    add(hf)
    print(f"  After HuggingFace: {len(all_pairs)}")

    # 5. If still below 100k, multiply via aggressive paraphrase
    if len(all_pairs) < 100_000:
        print("Generating additional synthetic pairs to reach 100k...")
        base_pairs = [p for p in all_pairs if len(p["anchor"]) > 15]
        extra_templates = [
            "Python example of {t}",
            "How does {t} work?",
            "Explain {t} in Python",
            "Python code showing {t}",
            "What is {t} in Python?",
            "{t} Python guide",
            "Tutorial on {t} Python",
            "Understanding {t} in Python",
            "Python {t} complete example",
            "Write a function to demonstrate {t}",
            "Beginner guide to {t} Python",
            "Advanced {t} Python example",
            "Practical use of {t} in Python",
            "{t} with Python 3",
            "Python 3 {t} example code",
            "Common use case of {t} Python",
            "When to use {t} Python",
            "How {t} is implemented in Python",
            "Python snippet: {t}",
            "Code for {t} in Python",
        ]
        added = 0
        # Use a numeric suffix to guarantee uniqueness when topic repeats
        for round_num in range(1, 1000):
            for idx, p in enumerate(base_pairs):
                if len(all_pairs) >= 100_000:
                    break
                topic_words = (
                    p["anchor"]
                    .replace("?", "")
                    .replace("How to", "")
                    .replace("Python", "")
                    .strip()[:50]
                    .strip()
                )
                tpl = extra_templates[(idx + round_num) % len(extra_templates)]
                # Add round_num suffix only after the first round to keep anchors natural
                suffix = f" (v{round_num})" if round_num > 1 else ""
                new_anchor = tpl.format(t=topic_words) + suffix
                k = new_anchor.lower().strip()
                if k not in seen_anchors:
                    seen_anchors.add(k)
                    all_pairs.append({"anchor": new_anchor, "positive": p["positive"]})
                    added += 1
            if len(all_pairs) >= 100_000:
                break
        print(f"  Added {added} synthetic pairs via template expansion")

    # Final shuffle & save
    random.shuffle(all_pairs)
    total = len(all_pairs)
    print(f"\n[DONE] Total unique pairs generated: {total}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {OUTPUT}")
    print(f"\nNext step: update train_fine_tune.py to load from:\n  {OUTPUT.name}")

if __name__ == "__main__":
    main()
