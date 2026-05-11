import torch 
import torch.nn as nn 
from utils.type import Tensor # torch.Tensor를 Tensor로 줄여서
from jaxtyping import Float
import torch.nn.functional as F

class DRNN_jm(nn.Module):
    def __init__(
        self,
        id: str, # DRNN
        max_delay: int, 
        init_tau: float,
        min_tau: float,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataset_id: str, 
        device: str,
        dropout: float = 0.2
    ):
        super().__init__() 
        self.max_delay: int = max_delay
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.num_train: int = 0
        self.device: str = device
        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.init_tau: float = init_tau 
        self.min_tau: float = min_tau
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.input_encoder = nn.Linear(self.input_size, 2 * self.hidden_size) 
        # self.h_encoder = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.m_encoder = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
    
        self.delay_layer = nn.Sequential(
             nn.Linear(self.hidden_size, self.hidden_size),
             nn.ReLU(),
             nn.Linear(self.hidden_size, self.max_delay))
         # k를 얼마나 delay할지 정하는 레이어. # 가중치행렬이 (1,md)
        
        self.output_layer = nn.Linear(
            self.hidden_size, self.output_size
        )
        
        if dataset_id in ["delaysequence"]: # forward 함수를 데이터셋에 맞게 바꿔줘
            self.forward = self.SignalPeriod_forward
        else:
            raise NotImplementedError(f"Dataset {dataset_id} not supported for RNN model.")
        
        self.to(self.device)
    
    @staticmethod
    def update_buffer(
        buffer: Float[Tensor, "b md h"], 
        m_t1: Float[Tensor, "b h 1"], 
        delay_steps: Float[Tensor, "b md h"], 
        train: bool
    ) -> Float[Tensor, "b md h"]:
        m_t1_t: Float[Tensor, "b 1 h"] = m_t1.transpose(1, 2) 
        new_buffer: Float[Tensor, "b md h"] = F.pad(buffer[:, 1:, :], (0, 0, 0, 1)) 
        values: Float[Tensor, "b md h"] = (m_t1_t * delay_steps) # broadcast to (b, md, h) 
        return new_buffer + values
        

    def SignalPeriod_forward(
            self,
            x: Tensor,
            train: bool,
            logs: dict | None = None

    ):
        bs, seq_len, _ = x.shape 
        current_tau = self.init_tau
        
        if train: 
            self.num_train += 1
            current_tau = max(self.min_tau, self.init_tau * (0.9999 ** self.num_train))
            if logs is not None:
                logs["model/tau"] = current_tau

        # # for Logging tensor
        # if logs is not None:
        #     delay_mean_log = x.new_zeros((bs, seq_len, 1))
        #     delay_std_log = x.new_zeros((bs, seq_len, 1))
            

        x_embed = self.input_encoder(x) 
        outputs = x.new_zeros((bs, seq_len, self.output_size)) 
        # h_t = x.new_zeros((bs, self.hidden_size))
        buffer = x.new_zeros((bs, self.max_delay, self.hidden_size))
        delay_records = x.new_zeros((bs, seq_len))

        for t in range(seq_len):
            m_t: Float[Tensor, "b h"] = self.tanh(buffer[:, 0, :]) 
            x_embed_t: Float[Tensor, "b 2h"] = x_embed[:, t, :] 
            # h_embed_t: Float[Tensor, "b 2h"] = self.h_encoder(h_t)

            m_emebed_t: Float[Tensor, "b 2h"] = self.m_encoder(m_t)
            emebed_t: Float[Tensor, "b 2h"] = x_embed_t + m_emebed_t
            
            raw_memory: Float[Tensor, "b h"] = self.tanh(emebed_t[:, self.hidden_size:])
            h_t: Float[Tensor, "b h"] = self.relu(emebed_t[:, :self.hidden_size])  
            m_t1: Float[Tensor, "b h 1"] = self.norm(raw_memory).unsqueeze(-1)  # 수정
            
            # 각각의 뉴런을 얼마나 delay할지
            shared_delay_logits = self.delay_layer(h_t) 
            delay_steps = shared_delay_logits.unsqueeze(1).expand(-1, self.hidden_size, -1)
            
            delay_one_hot: Float[Tensor, "b md h"] = F.gumbel_softmax(delay_steps, tau=current_tau, hard=True, dim=-1).transpose(1, 2) # forward에서 정수였던 애를 tau를 써서 부드러운 확률값으로 만들어 backward가 가능하게. (b,md,h)->(b,h,md)
            
            chosen_delay = delay_one_hot.argmax(dim=1).float().mean(dim=-1) # (b,h) 각 뉴런이 선택한 delay step
            delay_records[:, t] = chosen_delay.float() # delay 기록

            buffer =self.update_buffer(
                buffer, 
                m_t1, 
                delay_one_hot,
                train
            )
            
            h_t_dropped = self.dropout_layer(h_t)
            out = self.output_layer(h_t_dropped) # (b,h)-> (b,output)
            outputs[:, t, :] = out # t시점의 output 저장
            
            # if logs is not None:
            #     t_delays = delay_one_hot.transpose(1, 2).argmax(dim=-1).float()
            #     delay_mean_log[:, t, :] = t_delays.mean(dim=-1, keepdim=True)
            #     delay_std_log[:, t, :] = t_delays.std(dim=-1, keepdim=True)
                
        if logs is not None:
            logs["delay_records"] = delay_records
        return outputs, logs
    