"""
Generate a 500+ Python Code-Search Training Dataset in {"anchor": "...", "positive": "..."} format.
Targeted for fine-tuning SentenceTransformer with MultipleNegativesRankingLoss.
"""

import json
from pathlib import Path

DATASET = [
    # ── 1. GENERATORS, ITERATORS & ITERABLES ──────────────────────────
    {
        "anchor": "How to resume a function from where it left in python",
        "positive": "def resume_execution():\n    \"\"\"Generator function that pauses and resumes execution using yield.\"\"\"\n    print(\"Step 1\")\n    yield \"Paused at 1\"\n    print(\"Step 2\")\n    yield \"Paused at 2\"\n    print(\"Step 3\")"
    },
    {
        "anchor": "How to pause and resume function execution in Python",
        "positive": "def task_runner():\n    \"\"\"Pause execution with yield and resume with next().\"\"\"\n    print(\"Phase 1 complete\")\n    yield 1\n    print(\"Phase 2 complete\")\n    yield 2"
    },
    {
        "anchor": "How to create a simple generator function with yield in python",
        "positive": "def count_up_to(max_val):\n    \"\"\"Yield integers sequentially up to a maximum value.\"\"\"\n    count = 1\n    while count <= max_val:\n        yield count\n        count += 1"
    },
    {
        "anchor": "How does the yield keyword work in Python",
        "positive": "def simple_generator():\n    \"\"\"Yield returns a value and preserves function local state for next invocation.\"\"\"\n    yield \"First\"\n    yield \"Second\"\n    yield \"Third\""
    },
    {
        "anchor": "How to pass data back into a running generator using send method",
        "positive": "def receiver():\n    \"\"\"Receive values from caller using yield expression and .send().\"\"\"\n    while True:\n        val = yield\n        print(f\"Received: {val}\")\n\ng = receiver()\nnext(g)  # prime generator\ng.send(\"Hello\")"
    },
    {
        "anchor": "How to close a generator early in Python",
        "positive": "def stream_items():\n    try:\n        for i in range(100):\n            yield i\n    finally:\n        print(\"Generator cleanup executed\")\n\ng = stream_items()\nnext(g)\ng.close()"
    },
    {
        "anchor": "How to delegate iteration to another generator with yield from",
        "positive": "def sub_gen():\n    yield 1\n    yield 2\n\ndef main_gen():\n    \"\"\"Delegate iteration to sub_gen using yield from.\"\"\"\n    yield from sub_gen()\n    yield 3"
    },
    {
        "anchor": "How to flatten a nested list using recursive generator and yield from",
        "positive": "def flatten(nested):\n    \"\"\"Flatten arbitrarily nested iterables using recursion and yield from.\"\"\"\n    for item in nested:\n        if isinstance(item, (list, tuple)):\n            yield from flatten(item)\n        else:\n            yield item"
    },
    {
        "anchor": "How to implement a custom iterator class with __iter__ and __next__",
        "positive": "class RangeIterator:\n    \"\"\"Custom iterator class implementing __iter__ and __next__ dunder methods.\"\"\"\n    def __init__(self, start, stop):\n        self.current = start\n        self.stop = stop\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.current >= self.stop:\n            raise StopIteration\n        val = self.current\n        self.current += 1\n        return val"
    },
    {
        "anchor": "How to handle StopIteration in custom iterator in Python",
        "positive": "class FiniteIterator:\n    def __init__(self, limit):\n        self.n = 0\n        self.limit = limit\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.n >= self.limit:\n            raise StopIteration(\"Iteration limit reached\")\n        self.n += 1\n        return self.n"
    },
    {
        "anchor": "How to create an infinite generator in Python",
        "positive": "def infinite_sequence(start=0):\n    \"\"\"Generate an unbounded sequence of numbers.\"\"\"\n    num = start\n    while True:\n        yield num\n        num += 1"
    },
    {
        "anchor": "How to read huge files line by line lazily using generator",
        "positive": "def read_large_file(file_path):\n    \"\"\"Read file line by line without buffering entire file in RAM.\"\"\"\n    with open(file_path, 'r', encoding='utf-8') as f:\n        for line in f:\n            yield line.strip()"
    },
    {
        "anchor": "How to generate Fibonacci numbers with a generator in Python",
        "positive": "def fibonacci_gen():\n    \"\"\"Infinite Fibonacci number sequence generator.\"\"\"\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b"
    },
    {
        "anchor": "How to use itertools.cycle to loop infinitely over an iterable",
        "positive": "import itertools\n\ndef cycle_items(items, count):\n    \"\"\"Cycle repeatedly over items up to count iterations.\"\"\"\n    cycler = itertools.cycle(items)\n    return [next(cycler) for _ in range(count)]"
    },
    {
        "anchor": "How to chain multiple iterables together using itertools.chain",
        "positive": "import itertools\n\ndef combine_lists(list_a, list_b, list_c):\n    \"\"\"Chain multiple collections into a single continuous iterator.\"\"\"\n    return list(itertools.chain(list_a, list_b, list_c))"
    },
    {
        "anchor": "How to slice an iterator using itertools.islice",
        "positive": "import itertools\n\ndef get_first_n_items(iterable, n):\n    \"\"\"Extract first n items from an iterator without exhausting the rest.\"\"\"\n    return list(itertools.islice(iterable, n))"
    },
    {
        "anchor": "How to compute running cumulative sum using itertools.accumulate",
        "positive": "import itertools\n\ndef cumulative_sums(numbers):\n    \"\"\"Calculate running cumulative sums of a numerical sequence.\"\"\"\n    return list(itertools.accumulate(numbers))"
    },
    {
        "anchor": "How to group consecutive elements by key using itertools.groupby",
        "positive": "import itertools\n\ndef group_consecutive(data, key_func):\n    \"\"\"Group sorted data by specified key function.\"\"\"\n    return {k: list(g) for k, g in itertools.groupby(data, key=key_func)}"
    },
    {
        "anchor": "How to generate Cartesian product of lists using itertools.product",
        "positive": "import itertools\n\ndef cartesian_product(list1, list2):\n    \"\"\"Compute Cartesian product combinations between two lists.\"\"\"\n    return list(itertools.product(list1, list2))"
    },
    {
        "anchor": "How to generate combinations and permutations in Python",
        "positive": "import itertools\n\ndef find_combinations(items, r):\n    \"\"\"Return r-length combinations and permutations of elements.\"\"\"\n    combs = list(itertools.combinations(items, r))\n    perms = list(itertools.permutations(items, r))\n    return {'combinations': combs, 'permutations': perms}"
    },
]

# Write categories builder
def build_full_dataset():
    records = list(DATASET)
    
    # ── Category templates generator ──
    categories = [
        # (Category Name, Items)
        ("dunder_methods", [
            ("implement __str__ and __repr__ for custom class representation",
             "class User:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n    def __repr__(self):\n        return f\"User(name={self.name!r}, age={self.age!r})\"\n    def __str__(self):\n        return f\"{self.name} ({self.age} years old)\""),
            ("implement __eq__ and __hash__ to use object in set or as dict key",
             "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __eq__(self, other):\n        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)\n    def __hash__(self):\n        return hash((self.x, self.y))"),
            ("make class instance callable like a function using __call__",
             "class Adder:\n    def __init__(self, value):\n        self.value = value\n    def __call__(self, x):\n        return self.value + x\n\nadd_five = Adder(5)\nresult = add_five(10)  # 15"),
            ("implement subscript index access using __getitem__ and __setitem__",
             "class KeyValueStore:\n    def __init__(self):\n        self._store = {}\n    def __getitem__(self, key):\n        return self._store[key]\n    def __setitem__(self, key, value):\n        self._store[key] = value\n    def __delitem__(self, key):\n        del self._store[key]"),
            ("implement len protocol using __len__ method",
             "class CustomCollection:\n    def __init__(self, items):\n        self.items = list(items)\n    def __len__(self):\n        return len(self.items)"),
            ("implement custom context manager using __enter__ and __exit__",
             "class Timer:\n    import time\n    def __enter__(self):\n        self.start = self.time.perf_counter()\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.elapsed = self.time.perf_counter() - self.start\n        print(f\"Elapsed time: {self.elapsed:.4f}s\")\n        return False"),
            ("implement operator overloading for addition using __add__ and __radd__",
             "class Vector:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n    def __add__(self, other):\n        if isinstance(other, Vector):\n            return Vector(self.x + other.x, self.y + other.y)\n        return NotImplemented"),
            ("intercept attribute access using __getattr__ and __setattr__",
             "class DynamicAttributes:\n    def __init__(self):\n        self._data = {}\n    def __getattr__(self, name):\n        if name in self._data:\n            return self._data[name]\n        raise AttributeError(f\"No attribute {name}\")\n    def __setattr__(self, name, value):\n        if name == '_data':\n            super().__setattr__(name, value)\n        else:\n            self._data[name] = value"),
            ("implement singleton pattern or customize object creation using __new__",
             "class Singleton:\n    _instance = None\n    def __new__(cls, *args, **kwargs):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance"),
            ("intercept subclass inheritance using __init_subclass__",
             "class BaseHandler:\n    registry = {}\n    def __init_subclass__(cls, handler_type=None, **kwargs):\n        super().__init_subclass__(**kwargs)\n        if handler_type:\n            cls.registry[handler_type] = cls"),
            ("make object work with in operator using __contains__",
             "class WordFilter:\n    def __init__(self, blocked_words):\n        self.blocked = set(blocked_words)\n    def __contains__(self, word):\n        return word.lower() in self.blocked"),
            ("implement comparison operators __lt__ __gt__ __le__ __ge__ using functools.total_ordering",
             "from functools import total_ordering\n\n@total_ordering\nclass Student:\n    def __init__(self, name, grade):\n        self.name = name\n        self.grade = grade\n    def __eq__(self, other):\n        return self.grade == other.grade\n    def __lt__(self, other):\n        return self.grade < other.grade"),
            ("optimize memory footprint and prevent dynamic __dict__ using __slots__",
             "class FastPoint:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y"),
            ("customize object boolean evaluation using __bool__",
             "class Account:\n    def __init__(self, balance):\n        self.balance = balance\n    def __bool__(self):\n        return self.balance > 0"),
        ]),
        
        ("decorators_closures", [
            ("write a basic function execution timer decorator",
             "import time\nfrom functools import wraps\n\ndef timeit(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.perf_counter()\n        res = func(*args, **kwargs)\n        print(f\"{func.__name__} took {(time.perf_counter() - start):.4f}s\")\n        return res\n    return wrapper"),
            ("create a retry decorator with maximum attempts and sleep delay",
             "import time\nfrom functools import wraps\n\ndef retry(max_attempts=3, delay=1.0):\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            for attempt in range(1, max_attempts + 1):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as exc:\n                    if attempt == max_attempts:\n                        raise\n                    time.sleep(delay)\n        return wrapper\n    return decorator"),
            ("memoize function output with functools.lru_cache",
             "from functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef slow_fibonacci(n):\n    if n < 2:\n        return n\n    return slow_fibonacci(n - 1) + slow_fibonacci(n - 2)"),
            ("freeze or pre-bind function arguments using functools.partial",
             "from functools import partial\n\ndef send_email(to, subject, body, sender='noreply@example.com'):\n    print(f\"From: {sender} To: {to} Subject: {subject}\")\n\nsend_alert = partial(send_email, subject=\"SYSTEM ALERT\")"),
            ("create a closure that maintains state between calls using nonlocal",
             "def make_counter(start=0):\n    count = start\n    def step():\n        nonlocal count\n        count += 1\n        return count\n    return step"),
            ("write a rate limiter decorator allowing N calls per period",
             "import time\nfrom functools import wraps\n\ndef rate_limit(max_calls, period):\n    calls = []\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            now = time.time()\n            nonlocal calls\n            calls = [c for c in calls if now - c < period]\n            if len(calls) >= max_calls:\n                raise RuntimeError(\"Rate limit exceeded\")\n            calls.append(now)\n            return func(*args, **kwargs)\n        return wrapper\n    return decorator"),
            ("write a decorator that validates argument types at runtime",
             "from functools import wraps\n\ndef enforce_types(**expected_types):\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            for name, val in kwargs.items():\n                if name in expected_types and not isinstance(val, expected_types[name]):\n                    raise TypeError(f\"Arg {name} must be {expected_types[name]}\")\n            return func(*args, **kwargs)\n        return wrapper\n    return decorator"),
            ("implement singleton pattern with a class decorator",
             "def singleton(cls):\n    instances = {}\n    def get_instance(*args, **kwargs):\n        if cls not in instances:\n            instances[cls] = cls(*args, **kwargs)\n        return instances[cls]\n    return get_instance"),
            ("define a property with getter setter and deleter in Python",
             "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n    @property\n    def radius(self):\n        return self._radius\n    @radius.setter\n    def radius(self, val):\n        if val <= 0:\n            raise ValueError(\"Radius must be positive\")\n        self._radius = val\n    @radius.deleter\n    def radius(self):\n        del self._radius"),
            ("cache property computation on instance using functools.cached_property",
             "from functools import cached_property\n\nclass DatasetLoader:\n    def __init__(self, path):\n        self.path = path\n    @cached_property\n    def raw_data(self):\n        with open(self.path) as f:\n            return f.read()"),
        ]),

        ("asyncio_concurrency", [
            ("run multiple coroutines concurrently using asyncio.gather",
             "import asyncio\n\nasync def fetch_data(id, delay):\n    await asyncio.sleep(delay)\n    return f\"Data {id}\"\n\nasync def main():\n    results = await asyncio.gather(fetch_data(1, 0.2), fetch_data(2, 0.4))\n    return results"),
            ("run async tasks and process them as they complete using asyncio.as_completed",
             "import asyncio\n\nasync def worker(id):\n    await asyncio.sleep(id * 0.1)\n    return f\"Finished {id}\"\n\nasync def main():\n    tasks = [worker(i) for i in [3, 1, 2]]\n    for coro in asyncio.as_completed(tasks):\n        result = await coro\n        print(result)"),
            ("enforce timeout on an asynchronous operation using asyncio.timeout",
             "import asyncio\n\nasync def slow_task():\n    await asyncio.sleep(5)\n    return \"Done\"\n\nasync def main():\n    try:\n        async with asyncio.timeout(1.0):\n            await slow_task()\n    except TimeoutError:\n        print(\"Task timed out!\")"),
            ("limit concurrent async tasks using asyncio.Semaphore",
             "import asyncio\n\nasync def bounded_request(sem, url):\n    async with sem:\n        await asyncio.sleep(0.5)\n        return f\"Fetched {url}\"\n\nasync def main():\n    sem = asyncio.Semaphore(5)\n    tasks = [bounded_request(sem, f\"http://site.com/{i}\") for i in range(20)]\n    return await asyncio.gather(*tasks)"),
            ("create background task with fire and forget in asyncio",
             "import asyncio\n\nasync def log_activity(msg):\n    await asyncio.sleep(0.1)\n    print(f\"Logged: {msg}\")\n\nasync def handle_request():\n    asyncio.create_task(log_activity(\"Request received\"))\n    return {\"status\": \"ok\"}"),
            ("implement producer consumer queue in asyncio with asyncio.Queue",
             "import asyncio\n\nasync def producer(q):\n    for i in range(5):\n        await q.put(i)\n    await q.put(None)\n\nasync def consumer(q):\n    while True:\n        item = await q.get()\n        if item is None:\n            break\n        print(f\"Consumed {item}\")\n        q.task_done()"),
            ("run CPU bound tasks using concurrent.futures.ProcessPoolExecutor",
             "from concurrent.futures import ProcessPoolExecutor\n\ndef cpu_bound_fib(n):\n    if n < 2: return n\n    return cpu_bound_fib(n-1) + cpu_bound_fib(n-2)\n\nwith ProcessPoolExecutor() as executor:\n    results = list(executor.map(cpu_bound_fib, [30, 31, 32]))"),
            ("run I/O bound requests in parallel with ThreadPoolExecutor",
             "from concurrent.futures import ThreadPoolExecutor\nimport urllib.request\n\ndef fetch(url):\n    with urllib.request.urlopen(url) as r:\n        return r.status\n\nurls = ['https://httpbin.org/get'] * 5\nwith ThreadPoolExecutor(max_workers=5) as pool:\n    statuses = list(pool.map(fetch, urls))"),
            ("thread synchronization using threading.Lock",
             "import threading\n\nlock = threading.Lock()\ncounter = 0\n\ndef increment():\n    global counter\n    with lock:\n        counter += 1"),
            ("wait for thread signals using threading.Event",
             "import threading\n\nevent = threading.Event()\n\ndef worker():\n    print(\"Waiting for start signal...\")\n    event.wait()\n    print(\"Signal received, working!\")\n\nt = threading.Thread(target=worker)\nt.start()\nevent.set()"),
        ]),

        ("data_structures_collections", [
            ("count occurrences of items and find top N with collections.Counter",
             "from collections import Counter\n\nwords = ['apple', 'banana', 'apple', 'orange', 'apple', 'banana']\ncounts = Counter(words)\nmost_common = counts.most_common(2)  # [('apple', 3), ('banana', 2)]"),
            ("group elements without key error using collections.defaultdict",
             "from collections import defaultdict\n\nscores = defaultdict(list)\nscores['Alice'].append(95)\nscores['Bob'].append(88)"),
            ("create double ended queue with maxlen using collections.deque",
             "from collections import deque\n\nrecent = deque(maxlen=3)\nfor i in range(5):\n    recent.append(i)  # deque([2, 3, 4], maxlen=3)"),
            ("create lightweight tuple with named fields using collections.namedtuple",
             "from collections import namedtuple\n\nCoordinate = namedtuple('Coordinate', ['lat', 'lon'])\npt = Coordinate(lat=37.77, lon=-122.41)\nprint(pt.lat, pt.lon)"),
            ("combine multiple dictionaries without merging copies using collections.ChainMap",
             "from collections import ChainMap\n\ndefault_cfg = {'theme': 'light', 'timeout': 30}\nuser_cfg = {'timeout': 60}\ncfg = ChainMap(user_cfg, default_cfg)\nprint(cfg['timeout'])  # 60"),
            ("keep priority queue with min-heap operations using heapq",
             "import heapq\n\nnums = [5, 1, 9, 3, 7]\nheapq.heapify(nums)\nheapq.heappush(nums, 2)\nsmallest = heapq.heappop(nums)  # 1\ntop_three = heapq.nlargest(3, nums)"),
            ("binary search and maintain sorted list using bisect module",
             "import bisect\n\nsorted_list = [10, 20, 30, 40]\nidx = bisect.bisect_left(sorted_list, 25)  # 2\nbisect.insort(sorted_list, 25)  # [10, 20, 25, 30, 40]"),
            ("merge two dictionaries in Python 3.9+ using union operator",
             "dict1 = {'a': 1, 'b': 2}\ndict2 = {'b': 99, 'c': 3}\nmerged = dict1 | dict2  # {'a': 1, 'b': 99, 'c': 3}"),
            ("dictionary comprehension with filtering",
             "squares = {x: x**2 for x in range(10) if x % 2 == 0}"),
            ("set operations union intersection difference symmetric difference",
             "a = {1, 2, 3, 4}\nb = {3, 4, 5, 6}\nunion = a | b\nintersection = a & b\ndifference = a - b\nsym_diff = a ^ b"),
            ("invert a dictionary mapping values to keys",
             "original = {'a': 1, 'b': 2, 'c': 3}\ninverted = {val: key for key, val in original.items()}"),
            ("sort list of dictionaries by a specific key",
             "users = [{'name': 'Bob', 'age': 30}, {'name': 'Alice', 'age': 25}]\nsorted_users = sorted(users, key=lambda u: u['age'])"),
            ("find key with maximum value in dictionary",
             "scores = {'math': 88, 'physics': 95, 'chemistry': 91}\ntop_subject = max(scores, key=scores.get)"),
        ]),

        ("file_io_serialization", [
            ("read and write JSON file safely in Python",
             "import json\n\ndata = {'name': 'Alice', 'roles': ['admin', 'user']}\nwith open('data.json', 'w', encoding='utf-8') as f:\n    json.dump(data, f, indent=2)\n\nwith open('data.json', 'r', encoding='utf-8') as f:\n    loaded = json.load(f)"),
            ("read CSV into dictionary rows using csv.DictReader",
             "import csv\n\ndef read_csv(path):\n    with open(path, newline='', encoding='utf-8') as f:\n        return list(csv.DictReader(f))"),
            ("write list of dictionaries to CSV using csv.DictWriter",
             "import csv\n\ndef write_csv(path, rows, fieldnames):\n    with open(path, 'w', newline='', encoding='utf-8') as f:\n        writer = csv.DictWriter(f, fieldnames=fieldnames)\n        writer.writeheader()\n        writer.writerows(rows)"),
            ("use pathlib.Path for cross platform file path manipulations",
             "from pathlib import Path\n\nbase = Path('/var/log')\nlog_file = base / 'app' / 'server.log'\nlog_file.parent.mkdir(parents=True, exist_ok=True)\nif log_file.exists():\n    content = log_file.read_text(encoding='utf-8')"),
            ("recursively find all files matching glob pattern with pathlib",
             "from pathlib import Path\n\npy_files = list(Path('.').rglob('*.py'))"),
            ("safely write to temporary file using tempfile module",
             "import tempfile\n\nwith tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:\n    tmp.write('temporary data')\n    tmp.seek(0)\n    content = tmp.read()"),
            ("copy move and delete directory tree using shutil",
             "import shutil\n\nshutil.copytree('source_dir', 'backup_dir', dirs_exist_ok=True)\nshutil.rmtree('backup_dir')"),
            ("serialize and deserialize Python objects with pickle",
             "import pickle\n\nobj = {'model_weights': [0.1, 0.5, 0.9]}\nserialized = pickle.dumps(obj)\nrecovered = pickle.loads(serialized)"),
            ("read binary file chunk by chunk",
             "def read_binary(file_path, chunk_size=8192):\n    with open(file_path, 'rb') as f:\n        while chunk := f.read(chunk_size):\n            yield chunk"),
            ("zip and extract archive files using zipfile",
             "import zipfile\n\nwith zipfile.ZipFile('archive.zip', 'w') as z:\n    z.write('document.txt')\n\nwith zipfile.ZipFile('archive.zip', 'r') as z:\n    z.extractall('extracted_folder')"),
        ]),

        ("typing_dataclasses", [
            ("define modern typed dataclass with default values and factory",
             "from dataclasses import dataclass, field\n\n@dataclass\nclass Config:\n    host: str = 'localhost'\n    port: int = 8080\n    endpoints: list[str] = field(default_factory=list)"),
            ("define generic class with TypeVar in typing",
             "from typing import TypeVar, Generic\n\nT = TypeVar('T')\n\nclass Box(Generic[T]):\n    def __init__(self, item: T):\n        self.item = item\n    def get(self) -> T:\n        return self.item"),
            ("specify union and optional type annotations in Python 3.10+",
             "def parse_id(val: int | str | None) -> str:\n    if val is None:\n        return 'anonymous'\n    return str(val)"),
            ("define callback signature with typing.Callable",
             "from typing import Callable\n\ndef process_items(items: list[int], func: Callable[[int], int]) -> list[int]:\n    return [func(x) for x in items]"),
            ("define structural protocol for duck typing interface",
             "from typing import Protocol\n\nclass Drawable(Protocol):\n    def draw(self) -> None: ...\n\ndef render(obj: Drawable):\n    obj.draw()"),
            ("use Literal type to restrict parameter to set of exact strings",
             "from typing import Literal\n\ndef set_mode(mode: Literal['read', 'write', 'append']):\n    print(f\"Opening in {mode} mode\")"),
            ("use TypedDict for type checked dictionary structures",
             "from typing import TypedDict\n\nclass UserPayload(TypedDict):\n    username: str\n    email: str\n    is_admin: bool"),
            ("define Pydantic BaseModel with field validators",
             "from pydantic import BaseModel, Field, field_validator\n\nclass User(BaseModel):\n    id: int\n    email: str\n    age: int = Field(ge=0, le=120)\n    @field_validator('email')\n    def validate_email(cls, v):\n        if '@' not in v:\n            raise ValueError('Invalid email')\n        return v"),
        ]),

        ("algorithms_math", [
            ("implement binary search algorithm iteratively in Python",
             "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"),
            ("implement quicksort divide and conquer in Python",
             "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)"),
            ("implement two pointers two sum solution on sorted array",
             "def two_sum_sorted(nums, target):\n    i, j = 0, len(nums) - 1\n    while i < j:\n        s = nums[i] + nums[j]\n        if s == target: return [i, j]\n        elif s < target: i += 1\n        else: j -= 1\n    return []"),
            ("sliding window maximum sum subarray of size k",
             "def max_subarray_sum(nums, k):\n    if len(nums) < k: return 0\n    curr = sum(nums[:k])\n    max_val = curr\n    for i in range(k, len(nums)):\n        curr += nums[i] - nums[i - k]\n        max_val = max(max_val, curr)\n    return max_val"),
            ("breadth first search traversal on graph adjacency list",
             "from collections import deque\n\ndef bfs(graph, root):\n    visited = {root}\n    q = deque([root])\n    order = []\n    while q:\n        node = q.popleft()\n        order.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor)\n                q.append(neighbor)\n    return order"),
            ("depth first search traversal on graph with recursion",
             "def dfs(graph, node, visited=None):\n    if visited is None: visited = set()\n    visited.add(node)\n    path = [node]\n    for neighbor in graph.get(node, []):\n        if neighbor not in visited:\n            path.extend(dfs(graph, neighbor, visited))\n    return path"),
            ("detect cycle in directed graph using DFS and color marking",
             "def has_cycle(graph):\n    visited = {}  # 0: visiting, 1: visited\n    def dfs(node):\n        visited[node] = 0\n        for nxt in graph.get(node, []):\n            if visited.get(nxt) == 0: return True\n            if nxt not in visited and dfs(nxt): return True\n        visited[node] = 1\n        return False\n    return any(dfs(n) for n in graph if n not in visited)"),
            ("calculate greatest common divisor and least common multiple using math",
             "import math\n\ndef gcd_lcm(a, b):\n    g = math.gcd(a, b)\n    l = math.lcm(a, b)\n    return {'gcd': g, 'lcm': l}"),
            ("generate pseudo random numbers and sample choices using random module",
             "import random\n\ndef lottery(candidates, k=3):\n    random.seed(42)\n    chosen = random.sample(candidates, k)\n    random.shuffle(candidates)\n    return chosen"),
        ]),

        ("web_api_networking", [
            ("send HTTP GET and POST request with urllib.request",
             "import urllib.request\nimport json\n\ndef post_json(url, payload):\n    data = json.dumps(payload).encode('utf-8')\n    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})\n    with urllib.request.urlopen(req) as resp:\n        return json.loads(resp.read().decode())"),
            ("make async HTTP GET request with aiohttp",
             "import aiohttp\nimport asyncio\n\nasync def fetch_json(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()"),
            ("create FastAPI GET endpoint with query parameters and response model",
             "from fastapi import FastAPI, Query\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass User(BaseModel):\n    name: str\n    tier: str\n\n@app.get('/users', response_model=list[User])\ndef get_users(tier: str = Query('free', enum=['free', 'pro'])):\n    return [{'name': 'Alice', 'tier': tier}]"),
            ("parse query parameters and URL components with urllib.parse",
             "from urllib.parse import urlparse, parse_qs\n\nurl = 'https://api.example.com/search?q=python&page=2'\nparsed = urlparse(url)\nparams = parse_qs(parsed.query)  # {'q': ['python'], 'page': ['2']}"),
            ("create simple HTTP server using http.server in Python",
             "import http.server\nimport socketserver\n\nPORT = 8000\nHandler = http.server.SimpleHTTPRequestHandler\nwith socketserver.TCPServer(('', PORT), Handler) as httpd:\n    print(f'Serving on port {PORT}')\n    httpd.serve_forever()"),
        ]),

        ("exceptions_debugging_logging", [
            ("configure logging with rotating file handler",
             "import logging\nfrom logging.handlers import RotatingFileHandler\n\nlogger = logging.getLogger('app')\nlogger.setLevel(logging.INFO)\nhandler = RotatingFileHandler('app.log', maxBytes=1_000_000, backupCount=3)\nformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')\nhandler.setFormatter(formatter)\nlogger.addHandler(handler)"),
            ("format and inspect exception stack trace with traceback module",
             "import traceback\n\ntry:\n    1 / 0\nexcept ZeroDivisionError:\n    error_msg = traceback.format_exc()\n    print(f\"Captured traceback:\\n{error_msg}\")"),
            ("chain exceptions using raise from syntax in Python",
             "class AppError(Exception): pass\n\ntry:\n    val = int(\"invalid\")\nexcept ValueError as err:\n    raise AppError(\"Failed to parse configuration\") from err"),
            ("suppress specific exceptions safely using contextlib.suppress",
             "import contextlib\nimport os\n\nwith contextlib.suppress(FileNotFoundError):\n    os.remove('temporary_file_that_may_not_exist.tmp')"),
        ])
    ]

    # Expand each entry into multiple query paraphrases/variations (anchors)
    # Developers query the same code in 5-8 distinct ways
    paraphrase_templates = [
        "{query}",
        "How to {query} in Python",
        "Python code to {query}",
        "Example of how to {query} in Python",
        "Best practice to {query} with Python",
        "Explain and show code to {query}",
        "How do I {query}?"
    ]

    seen = set()
    for cat_name, items in categories:
        for query_desc, code_body in items:
            for tpl in paraphrase_templates:
                anchor = tpl.format(query=query_desc).strip()
                # capitalize first letter
                anchor = anchor[0].upper() + anchor[1:]
                key = (anchor, code_body)
                if key not in seen:
                    seen.add(key)
                    records.append({
                        "anchor": anchor,
                        "positive": code_body
                    })

    return records

if __name__ == "__main__":
    records = build_full_dataset()
    out_path = Path("data/python_code_search_500.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(records)} anchor-positive dataset pairs at {out_path}")

