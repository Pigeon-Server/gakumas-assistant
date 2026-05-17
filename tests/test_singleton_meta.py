import threading

from src.entity.Base import SingletonMeta


def test_singleton_meta_allows_nested_singleton_initialization():
    class Child(metaclass=SingletonMeta):
        pass

    class Parent(metaclass=SingletonMeta):
        def __init__(self):
            self.child = Child()

    result = {}

    def _build_parent():
        try:
            result["instance"] = Parent()
        except Exception as exc:  # pragma: no cover
            result["error"] = exc

    thread = threading.Thread(target=_build_parent, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert "error" not in result
    assert isinstance(result.get("instance"), Parent)
