# trace-light

[![License](https://img.shields.io/pypi/l/trace-light.svg?color=green)](https://github.com/kevinyamauchi/trace-light/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/trace-light.svg?color=green)](https://pypi.org/project/trace-light)
[![Python Version](https://img.shields.io/pypi/pyversions/trace-light.svg?color=green)](https://python.org)
[![CI](https://github.com/kevinyamauchi/trace-light/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinyamauchi/trace-light/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinyamauchi/trace-light/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinyamauchi/trace-light)

a light raytracing library

> This library is heavily inspired by and in part derived from
> [Optiland](https://github.com/optiland/optiland) (MIT License).
> See [NOTICE](NOTICE) for details.

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork kevinyamauchi/trace-light --clone
# or just
# gh repo clone kevinyamauchi/trace-light
cd trace-light
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
