from molgen.builders import opd_builder

class BuilderFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, class_name):
        self._builders[key] = class_name

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        return builder(**kwargs)

factory = BuilderFactory()
factory.register_builder('opd', opd_builder)
