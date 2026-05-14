from model.RNN import RNN
from model.LSTM import LSTM
from model.GRU import GRU
from model.DRNN_jm import DRNN_jm
from model.DRNN_v3 import DRNN_v3
from model.DRNN_sc import DRNN_sc

MODEL_REGISTRY = {
    "RNN": RNN,
    "LSTM": LSTM,
    "GRU": GRU,
    "DRNN_jm": DRNN_jm,
    "DRNN_v3": DRNN_v3,
    "DRNN_sc": DRNN_sc,
}