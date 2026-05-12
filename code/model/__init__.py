from model.RNN import RNN
from model.LSTM import LSTM
from model.GRU import GRU
from model.DRNN_jm import DRNN_jm

MODEL_REGISTRY = {
    "RNN": RNN,
    "LSTM": LSTM,
    "GRU": GRU,
    "DRNN_jm": DRNN_jm
}