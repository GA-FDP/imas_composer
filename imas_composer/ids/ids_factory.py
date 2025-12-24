"""
IDS Mapper Factory

This module provides a factory function to create the appropriate IDSMapper depending on 
tree selection
"""
from imas_composer.ids.base import IDSMapper
import importlib
import pkgutil
import pathlib
from typing import Dict, Type

def snake_to_pascal(snake_str: str) -> str:
    """Convert snake_case to PascalCase"""
    return ''.join(word.capitalize() for word in snake_str.split('_'))

class IDSFactory:

    IDS_list: Dict[str, Type[IDSMapper]] = {}

    def __init__(self):
        """
        Factory Class to create the appropriate mapper depending on tree choice.
        """
        package_name = "imas_composer.ids"
        package = importlib.import_module(package_name)
        package_path = pathlib.Path(package.__file__).parent
        for _, module_name, _  in pkgutil.iter_modules([str(package_path)]):
            if module_name in ["base", "ids_factory", "__init__"]:
                continue
            else:
                module = importlib.import_module(f"{package_name}.{module_name}")
                self.IDS_list[module_name] = getattr(module, snake_to_pascal(module_name)+"Mapper")
    
    def list_ids(self) -> list[str]:
        ids_list = []
        for ids_type in self.IDS_list:
            if "core_profiles" not in ids_type:
                ids_list.append(ids_type)
        ids_list.append("core_profiles")

        return ids_list

    def __call__(self, ids_type, **kwargs) -> IDSMapper:
        """
        Docstring for __call__
        
        :param ids_type: Which IDS to initialize
        :param kwargs: Keyword arguments to pass through the IDSMapper instance
        :return: IDSMapper instance
        :rtype: IDSMapper
        """
        if "core_profiles" not in ids_type:
            return self.IDS_list[ids_type](**kwargs)
        elif "ZIPFIT" in kwargs.get("profiles_tree", "ZIPFIT01") :
            return self.IDS_list[ids_type + "_zipfit"](**kwargs)
        elif kwargs.get("profiles_tree", "ZIPFIT01") == 'OMFIT_PROFS':
            return self.IDS_list[ids_type + "_omfit"](**kwargs)
        else:
            raise ValueError(
                f"Unknown profiles tree type: '{kwargs["profiles_tree"]}'. "
                f"Expected tree name containing 'ZIPFIT' or 'OMFIT_PROFS'"
        )