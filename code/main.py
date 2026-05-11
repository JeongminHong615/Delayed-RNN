import time
import wandb 
import hydra
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import torch.nn.utils.prune as prune
import seaborn as sns
import numpy as np
import torch.nn.functional as F
import matplotlib.patches as mpatches

from omegaconf import OmegaConf
from tqdm import tqdm 
from utils.type import hydra_dict
from utils.utils import set_seed, calculate_flops
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


from model import MODEL_REGISTRY
from dataset import DATASET_REGISTRY

plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False


def verify_delay_numerically(model, dataset, device, display_len=20, search_window=2):
    model.eval()
    
    fixed_seq = [9, 5, 2, 7] 
    seq_len = len(fixed_seq)
    
    k = dataset.k
    delay = dataset.delay
    total_len = dataset.total_len
    input_size = dataset.input_size
    
    seq_tensor = torch.tensor(fixed_seq, dtype=torch.long)
    input_seq = torch.zeros(total_len, input_size)
    
    input_seq[:seq_len, :k] = F.one_hot(seq_tensor, num_classes=k).float()
    
    target_delay = seq_len + delay
    input_seq[:, k+1] = target_delay / total_len
    start_output = target_delay
    input_seq[start_output-1, k] = 1.0 
    
    target_seq = torch.full((total_len,), -1, dtype=torch.long)
    target_seq[start_output:start_output + seq_len] = seq_tensor
    
    x_tensor = input_seq.unsqueeze(0).to(device)
    target_tensor = target_seq.unsqueeze(0).to(device)

    logs = {}
    with torch.no_grad():
        out, logs = model(x_tensor, train=False, logs=logs)
    
    x_np = x_tensor.squeeze(0).cpu().numpy()
    target_np = target_tensor.squeeze(0).cpu().numpy()
    out_probs = F.softmax(out.squeeze(0), dim=-1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))

    plot_x_np = x_np[:display_len, :k]
    plot_out_probs = out_probs[:display_len]
    plot_target_np = target_np[:display_len]

    sns.heatmap(plot_out_probs.T, ax=ax, cmap="Blues", cbar=False)

    input_times, input_classes = np.where(plot_x_np == 1.0)
    ax.scatter(input_times + 0.5, input_classes + 0.5, 
               color='lime', marker='s', s=150, edgecolor='black', linewidth=1.5, zorder=5, label='입력 데이터')

    valid_targets = plot_target_np != -1
    target_times = np.where(valid_targets)[0]
    target_classes = plot_target_np[valid_targets]
    ax.scatter(target_times + 0.5, target_classes + 0.5, 
               color='red', marker='x', s=100, linewidths=2.5, zorder=5, label='정답 데이터')

    if start_output - 1 < display_len:
        ax.axvline(x=start_output-1, color='red', linestyle='--', linewidth=2, label='구분자')

    has_success, has_failure = False, False

    for t_in in range(seq_len):
        c = int(fixed_seq[t_in])
        t_expected = start_output + t_in
        
        search_window_probs = out_probs[start_output:display_len, c]
        
        if len(search_window_probs) > 0:
            best_idx = np.argmax(search_window_probs)
            t_actual = start_output + best_idx
            
            if t_actual == t_expected:
                # 지연 성공
                arrow_color, arrow_style, lw, ls = "purple", "-|>", 3.0, "-"
                has_success = True
            else:
                # 지연 실패
                arrow_color, arrow_style, lw, ls = "orange", "->", 2.5, "--"
                has_failure = True
                
            tail_x = t_in + 0.8
            head_x = t_actual + 0.2
            y_pos = c + 0.5
            
            ax.annotate('', xy=(head_x, y_pos), xytext=(tail_x, y_pos),
                arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, lw=lw, linestyle=ls, alpha=0.9))

    handles, labels = ax.get_legend_handles_labels()
    if has_success:
        handles.append(mpatches.Patch(color='purple', label='지연 성공'))
    if has_failure:
        handles.append(mpatches.Patch(color='orange', label='지연 실패', linestyle='--'))
        
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=12)

    ax.set_ylabel("클래스 (0~k-1)", fontsize=12, fontweight='bold')
    ax.set_xlabel("타임스텝", fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    save_path = "delay_analysis.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300) 
    plt.close(fig)

@hydra.main(config_path="config", config_name="config", version_base=None) # config.yaml을 읽어와
def main(args: hydra_dict) -> None: # config.yaml을 args로 저장
    print("-"*20, "Experiment Configuration", "-"*20)
    print(OmegaConf.to_yaml(args)) # 실험세팅 텍스트 형태로 알려줘
    print("-"*60)
    
    seed: int = args.seed
    set_seed(seed = seed) # seed 고정 : 항상 같은 시작점에서 학습 # utils/utils.py
    
    num_epochs: int = args.num_epochs
    batch_size: int = args.batch_size
    hidden_size: int = args.hidden_size
    
    lr: float = args.lr
    
    wandb_args: hydra_dict = args.wandb
    dataset_args: hydra_dict = args.dataset
    model_args: hydra_dict = args.model
    
    use_wandb: bool = wandb_args.use_wandb # False
    use_model_compile: bool = args.use_model_compile # True
    use_lr_scheduler: bool = args.use_lr_scheduler # False
    
    dataset_id: str = dataset_args.id
    model_id: str = model_args.id
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
    train_dataset = DATASET_REGISTRY[dataset_id]( # init 파일 통해 -> dataset
        **dataset_args, # unpacking 딕셔너리 벗겨내
        train = True
    )
    eval_dataset = DATASET_REGISTRY[dataset_id](
        **dataset_args,
        train = False
    )
    
    input_size: int = train_dataset.input_size
    output_size: int = train_dataset.output_size
    seq_len: int = train_dataset.seq_len
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) 
    model = MODEL_REGISTRY[model_id](
        **model_args,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dataset_id=dataset_id,
        device=device
    )
    
    model.eval()

    num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Calculating Flops...") # 연산량 측정
    flops_million = calculate_flops( # utils/utils.py
        model=model,
        input_size=input_size,
        seq_len=seq_len,
        device=device
    ) 


    if args.get("only_flops", False): 
        print(f"Model: {model_id}")
        print(f"FLOPs (Million): {flops_million:.2f} M")
        print(f"✅ Parameters: {num_params:,} 개")
        sys.exit(0)
    
    if use_model_compile:
        print("Model compiling...")
        model = torch.compile(
            model = model,
            mode="reduce-overhead",
            fullgraph=True,
            disable=use_model_compile # ?
        )
        start_time = time.time()
        eval_dataset.init_model_compile(
            model,
            eval_dataloader,
            device,
        )
    
    compile_time = time.time() - start_time if use_model_compile else 0.0
    num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad) # 사용한 neuron 수 세기
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if use_lr_scheduler:
        warmup_epochs = 10 if num_epochs > 15 else max(1, num_epochs // 5)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=0.1*lr)
    
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    if use_wandb: 
        group_name: str = wandb_args.group_name
        unique_id = f"{model_id}_{group_name}_{dataset_id}_{seed}_{time.time()}" # 실험 이름 설정
        wandb.init(
            entity=args.wandb.entity,
            project=dataset_id, 
            id=unique_id,
            name=f"{model_id}_{seed}",
            group=group_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        
        wandb.config.update({"Flops(1M)": flops_million})
        wandb.config.update({"num_params": num_params})
        wandb.config.update({"compile_time": compile_time if use_model_compile else None}) #

    
    print(f"Dataset: {dataset_id} | Model: {model_id} | FLOPs(M): {flops_million:.2f} | Compile Time(s): {compile_time:.2f} | Number of Parameters: {num_params}")
    best_loss = float('inf')
    saved_model_path = f"best_model_{model_id}.pth"
    

    for epoch in tqdm(range(num_epochs), desc=f"Epochs [{dataset_id}]"):
        train_logs = train_dataset.train(
            model = model,
            dataloader = train_dataloader,
            optimizer = optimizer,
            device = device,
        )
        eval_logs = eval_dataset.evaluate(
            model = model,
            dataloader = eval_dataloader,
            device = device,
        )
        if use_lr_scheduler: # 한 epoch 끝날때마다 lr 조절
            scheduler.step()
        if use_wandb:
            
            wandb.log({
                **train_logs,
                **eval_logs
            })  
        
        current_loss = eval_logs.get('Eval/loss', float('inf'))
        
        if current_loss < best_loss:
            best_loss = current_loss
            # 모델의 가중치(state_dict)를 파일로 안전하게 저장합니다!
            torch.save(model.state_dict(), saved_model_path)
            # print(f"  --> [Epoch {epoch}] 최고 성능 갱신! 모델이 저장되었습니다. (Loss: {best_loss:.4f})")
    
    print(f"\n 학습 종료! 최고 성능 모델({saved_model_path})을 불러와 수치 검증을 시작합니다.")
    model.load_state_dict(torch.load(saved_model_path, map_location=device, weights_only=True))

    verify_delay_numerically(model, eval_dataset, device)    



if __name__ == '__main__': 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 설정된 장치: {device}")
    main()
