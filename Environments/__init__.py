import importlib.util
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name + '.py')

    # Create a module spec
    spec = importlib.util.spec_from_file_location(name, pathname)

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    return module
