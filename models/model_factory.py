from models.conv1d_model import Conv1DModel
# from models.conv1d_td_model import Conv1DTimeDistributedModel
# from models.conv1d_td_lstm_model import Conv1DTimeDistributedLSTMModel
# from models.conv2d_td_lstm_model import Conv2DTimeDistributedLSTMModel

class ModelFactory:
    @staticmethod
    def create(model_type, input_shape, num_classes):
        model_map = {
            "conv1d": Conv1DModel(),
            # "conv1d_td": Conv1DTimeDistributedModel(),
            # "conv1d_td_lstm": Conv1DTimeDistributedLSTMModel(),
            # "conv2d_td_lstm": Conv2DTimeDistributedLSTMModel(),
        }
        return model_map[model_type].build(input_shape, num_classes)
