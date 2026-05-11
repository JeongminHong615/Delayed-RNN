import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset 
import torch.nn.functional as F

class DelaySequenceDataset(Dataset):
    def __init__(
            self,
            id,
            size, # 총 데이터 개수
            min_len=5, # 데이터 하나의 길이
            max_len=100,
            k = 10, # 클래스 개수(one-hot vector) 종류 -> 그니까 수는 0~9까지만 가능
            train=True
        ):
        super().__init__()
        self.id = id
        self.size = size
        self.min_len = min_len
        self.max_len = max_len
        self.k = k
        self.min_delay = 5
        self.max_delay = 50
        self.train_mode = train

        self.input_size = k+2 # one-hot vector + 구분자 + delay 토큰
        self.total_len = max_len + self.max_delay + max_len # 입력 시퀀스의 총 길이 (최대 시퀀스 길이 + 최대 지연 + 최대 시퀀스 길이)
        self.output_size = k
        self.seq_len = self.total_len


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq_len = random.randint(self.min_len, self.max_len)
        delay = random.randint(self.min_delay, self.max_delay)

        seq = [random.randint(0, self.k - 1) for _ in range(seq_len)] # seq_len개의 데이터(0~k의 정수) 뽑기
        seq_tensor = torch.tensor(seq, dtype=torch.long) # 정수 시퀀스를 텐서로 변환

        input_seq = torch.zeros(self.total_len, self.input_size) # 입력 시퀀스 초기화
        input_seq[:seq_len, :self.k] = F.one_hot(seq_tensor, num_classes=self.k).float()

        input_seq[seq_len-1, self.k] = 1.0 # 입력 끝나는 시점 구분자

        start_output = seq_len + delay
        input_seq[start_output - 1, self.k + 1] = 1.0 # 출력 시작하는 시점 구분자

        target_seq = torch.full((self.total_len,), -1, dtype=torch.long)
        target_seq[start_output:start_output + seq_len] = seq_tensor

        return input_seq, target_seq

    @staticmethod
    def init_model_compile(
        model,
        dataloader,
        device
    ):
        print("Warming up compiled model...")
        model.eval()
        
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.no_grad():
                model(x=inputs, train=False, logs={})
            break 
        
    @staticmethod
    def train(
        model,
        dataloader,
        optimizer,
        device
    ):
        model.train()
        model_log_dict: dict[str, list[float]] = {}
        
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            out, model_logs = model(x=inputs, train=True, logs={})

            loss_logs = compute_loss(
                out = out,
                targets = targets,
                model = model,
                optimizer = optimizer
            )
            model_log_dict = logging(
                model_log_dict=model_log_dict, 
                logs={**model_logs, **loss_logs}
            )
        return remove_list_log(model_log_dict, section = "Train")
    
    @staticmethod
    def evaluate(
        model,
        dataloader,
        device
        ):
        model.eval()
        model_log_dict: dict[str, list[float]] = {}

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            out, model_logs = model(x=inputs, train=False, logs={})

            loss_logs = compute_loss(
                out = out,
                model = model,
                targets = targets
            )
            model_log_dict = logging(
                model_log_dict=model_log_dict, 
                logs={**model_logs, **loss_logs}
            )
        return remove_list_log(model_log_dict, section= "Eval")

def compute_loss(
        out,
        model,
        targets,
        inputs=None,
        model_logs=None,
        optimizer=None
        ):
    bs, seq_len, k = out.size()

    mask = targets != -1
    out_valid = out[mask]
    targets_valid = targets[mask]

    loss_fn = nn.CrossEntropyLoss()
    loss_se = loss_fn(out_valid, targets_valid)
    loss = loss_se

    # delay 학습
    loss_delay = torch.tensor(0.0, device=out.device)
    if inputs is not None and model_logs is not None and "delay_logits" in model_logs:
        delay_logits = model_logs["delay_logits"] 
        
        hint_idx = inputs[:, :, k+1].argmax(dim=1)  
        end_idx = inputs[:, :, k].argmax(dim=1)     
        true_delay_val = hint_idx - end_idx      
        
        target_delay_val = true_delay_val.unsqueeze(1).expand(-1, seq_len)
        
        input_mask = inputs[:, :, :k].sum(dim=-1) > 0
        
        if input_mask.any():
            delay_logits_valid = delay_logits[input_mask]
            target_delay_valid = target_delay_val[input_mask]
            
            target_delay_valid = torch.clamp(target_delay_valid, max=delay_logits.size(-1) - 1)
            
            delay_loss_fn = nn.CrossEntropyLoss()
            loss_delay = delay_loss_fn(delay_logits_valid, target_delay_valid)
            
            loss = loss_se + loss_delay

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    with torch.inference_mode():
        logs = {}
        
        # chracter-level accuracy
        valid_preds = out_valid.argmax(dim=-1)
        acc = (valid_preds == targets_valid).float().mean()
        
        full_preds = out.argmax(dim=-1) 
        
        correct_matrix = (full_preds == targets) 
        
        ignore_mask = (targets == -1)
        match_matrix = correct_matrix | ignore_mask 
        
        seq_correct = match_matrix.all(dim=1) 
        
        seq_acc = seq_correct.float().mean()
        preds = F.one_hot(valid_preds, num_classes=k).float()
        targets_oh = F.one_hot(targets_valid, num_classes=k).float() 

        TP = (preds * targets_oh).sum(dim=0)
        FP = (preds * (1 - targets_oh)).sum(dim=0)
        FN = ((1 - preds) * targets_oh).sum(dim=0)
        eps = 1e-7
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        logs["f1_score"] = macro_f1.item()
        logs["precision"] = macro_precision.item()
        logs["recall"] = macro_recall.item()
        logs["accuracy"] = acc.item()            
        logs["seq_accuracy"] = seq_acc.item()   
        logs["loss"] = loss.item()

    return logs

def logging(
    model_log_dict: dict, 
    logs: dict
):
    for log_key, log_value in logs.items():
        if log_key in ["delay_records", "delay_logits_records"]:
            continue

        if log_key not in model_log_dict:
            model_log_dict[log_key] = []

        if isinstance(log_value, torch.Tensor):
            if log_value.numel() == 1:
                model_log_dict[log_key].append(log_value.item())

        elif isinstance(log_value, (float, int)):
            model_log_dict[log_key].append(log_value)
            
        else:
            raise ValueError(f"Unsupported log value type: {type(log_value)} for key: {log_key}")
    return model_log_dict

def remove_list_log(
    model_log_dict: dict,
    section: str
):
    log_dict = {}
    for key, values in model_log_dict.items():
        if isinstance(values[0], (float, int)):
            log_dict[f"{section}/{key}"] = sum(values) / len(values)
        else:
            raise ValueError(f"Unsupported log value type in model_log_dict for key: {key}")
    return log_dict
      