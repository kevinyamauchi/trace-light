import trace_light


def test_imports_with_version():
    assert isinstance(trace_light.__version__, str)
