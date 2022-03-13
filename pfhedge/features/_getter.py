from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Tuple
from typing import Type
from typing import Union

from ._base import Feature


class FeatureFactory:

    _features: Dict[str, Type[Feature]]

    # singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance._features = OrderedDict()
        return cls._instance

    def register_feature(self, name: str, cls: Type[Feature]) -> None:
        """Adds a feature to the factory.

        Args:
            name (str): name of the feature.
            cls (type(Feature)): feature class to be registered.
        """
        self._features[name] = cls

    def named_features(self) -> Iterator[Tuple[str, Type[Feature]]]:
        """Returns an iterator over feature classes, yielding both the
        name of the feature class as well as the feature class itself.

        Yields:
            (string, type(Feature)): Tuple containing
                the name and feature class.
        """
        for name, feature in self._features.items():
            if feature is not None:
                yield name, feature

    def names(self) -> Iterator[str]:
        """Returns an iterator over the names of the feature classes.

        Yields:
            str: name of the feature class.
        """
        for name, _ in self.named_features():
            yield name

    def features(self) -> Iterator[Type[Feature]]:
        """Returns an iterator over feature classes.

        Yields:
            type(Feature): Feature class.
        """
        for _, feature in self.named_features():
            yield feature

    def get_class(self, name: str) -> Type[Feature]:
        """Returns the feature class with the given name.

        Parameters:
            name (str): name of the feature class.

        Returns:
            type(Feature): feature class.
        """
        if name not in self.names():
            raise KeyError(
                f"{name} is not a valid name. "
                "Use pfhedge.features.list_feature_names() to see available names."
            )
        return self._features[name]

    def get_instance(self, name: str, **kwargs: Any) -> Feature:
        """Returns the feature with the given name.

        Parameters:
            name (str): name of the feature class.

        Returns:
            Feature: feature.
        """
        return self.get_class(name)(**kwargs)  # type: ignore


def get_feature(feature: Union[str, Feature], **kwargs: Any) -> Feature:
    """Get feature from name.

    Args:
        name (str): Name of feature.
        **kwargs: Keyword arguments to pass to feature constructor.

    Returns:
        Feature

    Examples:
        >>> from pfhedge.features import get_feature
        ...
        >>> get_feature("moneyness")
        <pfhedge.features.features.Moneyness object at ...>
    """
    if isinstance(feature, str):
        feature = FeatureFactory().get_instance(feature, **kwargs)
    elif not isinstance(feature, Feature):
        raise TypeError(f"{feature} is not an instance of Feature.")
    return feature


def list_feature_dict() -> dict:
    return dict(FeatureFactory().named_features())


def list_feature_names() -> list:
    """Returns the list of the names of available features.

    Returns:
        list[str]
    """
    return sorted(list(FeatureFactory().names()))


def list_features() -> list:
    return list(FeatureFactory().features())
