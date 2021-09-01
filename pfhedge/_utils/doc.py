def _set_docstring(object: object, name: str, value: object) -> None:
    # so that object.name.__doc__ == value.__doc__
    setattr(getattr(object, name), "__doc__", value.__doc__)


def _set_attr_and_docstring(object: object, name: str, value: object) -> None:
    setattr(object, name, value)
    _set_docstring(object, name, value)
