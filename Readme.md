# LD-RNN

가상 환경 활성화
source myenv/bin/activate

### 쉘 스크립트를 사용하여 실험한다.

bash delaysequence.sh [GPU_ID] [MODEL] [DATASET] [HIDDEN_SIZE] [USE_WANDB] [GROUP_NAME]

### Structure

DelayRNN/
├── code/
│ ├── dataset/
│ │ └── delaysequence.py
│ ├── models/
│ │ └── DRNN_jm.py  
│ ├── utils/
│ │ └── utils.py  
│ ├── scripts/
│ │ └── delaysequence.sh
│ └ main.py  
└── README.md
