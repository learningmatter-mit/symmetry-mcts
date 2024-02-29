from environments.builders import y6_builder, patent_builder

class BuilderFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, class_name):
        self._builders[key] = class_name

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        return builder(**kwargs)

factory = BuilderFactory()
factory.register_builder('y6', y6_builder)
factory.register_builder('patent', patent_builder)
