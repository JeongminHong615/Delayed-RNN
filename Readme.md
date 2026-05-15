# LD-RNN: Learnable Axonal Delay RNN

KCC 2026 논문 "LD-RNN: 학습 가능한 축삭 지연을 활용한 순환 신경망"의 공식 구현체입니다.

## 개요

LD-RNN은 어텐션의 사후 조회(retrospective retrieval)나 LSTM의 손실 압축(lossy compression)과 구분되는 **push-to-future** 메커니즘을 표준 RNN에 도입한 모델입니다. 학습 가능한 축삭 지연(axonal delay)을 통해 과거 정보를 학습된 지연만큼 미래의 특정 시점으로 직접 송신하며, 이산적 지연 변수는 Straight-Through Gumbel-Softmax로 역전파 학습됩니다.

DelaySequence(가변 동적 지연 복사 과제) 실험에서 LD-RNN은 LSTM 대비 약 1/3, Transformer 대비 약 1/8의 매개변수와 1/10의 연산량으로 시퀀스 길이 N=50, 동적 지연 d∈{5,…,20} 조건에서 시퀀스 단위 정확도(SA)를 의미 있게 확보한 유일한 모델입니다.

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU 환경(CUDA)에서 검증되었습니다. `requirements.txt`는 PyTorch CUDA 12.8 빌드를 가정합니다.

## 디렉토리 구조

```
Delayed-RNN/
├── code/
│   ├── main.py                       # 학습/평가 진입점 (Hydra)
│   ├── plot_dynamic_delay.py         # Figure 2 (동적 지연 정성 분석) 생성
│   ├── config/
│   │   ├── config.yaml               # 기본 설정 (seed, lr, optimizer)
│   │   ├── dataset/delaysequence.yaml
│   │   └── model/{RNN,GRU,LSTM,Transformer,DRNN_jm}.yaml
│   ├── dataset/
│   │   └── delaysequence.py          # 동적 지연 복사 데이터셋
│   ├── model/
│   │   ├── DRNN_jm.py                # LD-RNN (제안 모델)
│   │   ├── LSTM.py, GRU.py, RNN.py   # 베이스라인
│   │   └── Transformer.py            # Transformer 인코더 베이스라인
│   ├── utils/                        # 시드 고정, FLOPs 계산
│   └── scripts/run_paper_experiments.sh
└── Readme.md
```

## 논문 실험 재현

### 한 줄로 전체 재현

```bash
cd code
bash scripts/run_paper_experiments.sh 0   # GPU 0번 사용
```

위 스크립트는 RNN → GRU → LSTM → Transformer → LD-RNN 순서로 다섯 개 모델을 동일 조건에서 학습하고, 표 1과 동일한 지표(CA, SA, #Params, MFLOPS)를 출력합니다.

### 개별 모델 실행

각 모델은 `python main.py model=<MODEL>` 으로 단독 실행할 수 있습니다. 본 논문이 보고한 설정은 다음과 같습니다.

```bash
# 공통 인자
COMMON="dataset=delaysequence seed=0 num_epochs=50 hidden_size=64 lr=0.01 \
        dataset.min_len=50 dataset.max_len=50 \
        +dataset.min_delay=5 +dataset.max_delay=20"

# LSTM
CUDA_VISIBLE_DEVICES=0 python main.py model=LSTM $COMMON

# Transformer (L=4가 가장 성능이 좋아 표 1에 대표값으로 보고됨)
CUDA_VISIBLE_DEVICES=0 python main.py model=Transformer $COMMON \
    model.num_layers=4 model.num_heads=4 model.dim_feedforward=128

# LD-RNN (제안 모델)
CUDA_VISIBLE_DEVICES=0 python main.py model=DRNN_jm $COMMON \
    model.max_delay=75
```

학습이 끝나면 `best_model_<MODEL>.pth` 가 현재 디렉토리에 저장되고, 최종 에포크의 CA/SA가 로그 마지막 줄에 출력됩니다.

### 실험 설정 요약

| 항목 | 값 |
|-----|----|
| 입력 시퀀스 길이 $N$ | 50 (고정) |
| 클래스 수 $K$ | 10 |
| 동적 지연 범위 $d$ | Uniform{5, 6, …, 20} |
| 은닉 차원 | 64 |
| 옵티마이저 | Adam |
| 학습률 | 0.01 (warm-up + cosine 스케줄) |
| 배치 크기 | 128 |
| 에포크 | 50 |
| 손실 함수 | Cross-Entropy (마스킹 적용) |

## 표 1 결과 (논문 보고치)

시퀀스 길이 N=50, 동적 지연 d∈{5,…,20}, 단일 시드.

| 모델 | CA | SA | #Params | MFLOPS |
|-----|-----|-----|---------|--------|
| LSTM | 0.4971 | 0.0000 | 53,898 | 11.63 |
| Transformer (L=4) | 0.9563 | 0.0140 | 148,170 | 39.74 |
| **LD-RNN (제안)** | **0.9469** | **0.6317** | **19,797** | **4.62** |

LD-RNN은 LSTM 대비 약 1/3, Transformer 대비 약 1/8 매개변수만으로 시퀀스 단위 완벽 복원에 성공한 유일한 모델입니다.

## 동적 지연 정성 분석 그림 재현

LD-RNN 학습이 끝나 `best_model_DRNN_jm.pth` 가 존재하는 상태에서:

```bash
cd code
python plot_dynamic_delay.py
```

동일 입력 시퀀스에 대해 지연 토큰만 d=5,6,7,8로 달리 주었을 때 LD-RNN이 각 정답 위치에서 정확히 출력함을 보여주는 4-panel 그림(`delay_analysis_dynamic.png`)이 생성됩니다.

## 인용

```bibtex
@inproceedings{hong2026ldrnn,
  title  = {LD-RNN: 학습 가능한 축삭 지연을 활용한 순환 신경망},
  author = {홍정민 and 박현우 and 성백륜 and 고상기},
  booktitle = {Proc. KCC},
  year   = {2026}
}
```
