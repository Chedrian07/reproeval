"""Canned programs used by sandbox-related tests.

Each constant is a complete, self-contained Python program suitable for
feeding into ``SandboxRunner.execute()``.
"""

# Prints a single well-known line -- the simplest "it works" smoke test.
SIMPLE_HELLO: str = 'print("hello world")\n'

# Defines a function, calls it, and prints the result.
SIMPLE_ADD: str = """\
def add(a, b):
    return a + b

print(add(2, 3))
"""

# Runs forever.  Used to verify timeout enforcement.
INFINITE_LOOP: str = "while True:\n    pass\n"

# Contains a syntax error -- should produce a non-zero exit code and stderr.
SYNTAX_ERROR: str = "def foo(:\n    pass\n"

# Produces ~1 MB of output.  Used to verify output truncation.
LARGE_OUTPUT: str = 'print("x" * 1_000_000)\n'

# Writes to stderr via sys.stderr.
STDERR_OUTPUT: str = """\
import sys
print("stdout line", flush=True)
print("stderr line", file=sys.stderr, flush=True)
"""

# Exits with a specific non-zero code.
EXIT_CODE_42: str = """\
import sys
sys.exit(42)
"""
