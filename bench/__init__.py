"""Benchmark harness for the minillm serving engine.

Public entry points:
- ``python -m bench.run --config <toml>``: execute one run and write a self-contained run dir.
- ``bench.index``: validate runs and generate ``results/INDEX.md``.
- ``bench.metrics`` / ``bench.loadgen`` / ``bench.engines``: building blocks.
"""

__version__ = "0.1.0"
