from models.conv1d_model import Conv1DModel
from models.conv1d_td_model import Conv1DTimeDistributedModel
from models.conv1d_td_lstm_model import Conv1DTimeDistributedLSTMModel
from models.conv2d_td_lstm_model import Conv2DTimeDistributedLSTMModel
from models.conv2d_model import Conv2DModel
from models.conv2d_td_model import Conv2DTimeDistributedModel
from models.conv2d_td_lstm_bd_model import Conv2DTimeDistributedLSTMBDModel

class ModelFactory:
    @staticmethod
    def create(model_type, input_shape, num_classes):
        model_map = {
            "conv1d": Conv1DModel(),
            "conv1d_td": Conv1DTimeDistributedModel(),
            "conv1d_td_lstm": Conv1DTimeDistributedLSTMModel(),
            "conv2d_td_lstm": Conv2DTimeDistributedLSTMModel(),
            "conv2d": Conv2DModel(),
            "conv2d_td": Conv2DTimeDistributedModel(),
            "conv2d_td_lstm_bd" : Conv2DTimeDistributedLSTMBDModel()
        }
        return model_map[model_type].build(input_shape, num_classes)
