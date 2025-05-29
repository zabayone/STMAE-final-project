from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def build(self, input_shape, num_classes):
        pass