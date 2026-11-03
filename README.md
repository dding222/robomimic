# Agibot Toy2Box — BC-RNN Observation Ablation

Isaac Lab v2.3.2의 **Agibot A2D Toy2Box** 작업에서 BC-RNN 정책을 학습하고, 관측 정보(Observation) 구성에 따른 물체 조작 성공률을 비교했다. 이미지 없이 로봇과 물체의 상태 정보만 사용했으며, 23차원 기본 관측에 상대 위치 또는 관절 상태를 추가했을 때의 성능을 평가했다.

## 1. 실험 구성

| 모델 구성 | Observation | 차원 |
|---|---|---:|
| **Base** | EEF Pose + Gripper + Toy Pose + Box Pose | **23-D** |
| **Base + Relative Position** | Base + EEF→Toy 상대 위치 XYZ | **26-D** |
| **Base + Joint State** | Base + Joint Position 34-D + Joint Velocity 34-D | **91-D** |

각 정책은 시연 데이터에서 동일한 형태의 7차원 End-Effector 제어 명령을 학습한다. 평가 지표는 Toy2Box 작업 성공률이며, 일부 모델은 에피소드 길이도 함께 기록했다.

## 2. 환경 및 데이터

### Task

| 항목 | 설정 |
|---|---|
| Simulator | Isaac Lab v2.3.2 / Isaac Sim |
| Robot | Agibot A2D, 오른팔 |
| Base task | `Agibot-Toy2Box-RmpFlow-v0` |
| Mimic task | `Agibot-Toy2Box-RmpFlow-Mimic-v0` |
| Object | ToyTruck, Box |
| Control | RMPFlow 기반 상대 EEF Pose 및 Gripper 제어 |
| Action | 7-D: `[dx, dy, dz, rx, ry, rz, gripper]` |
| Randomization | ToyTruck과 Box의 XY 위치 및 yaw |
| Success condition | ToyTruck이 Box의 위치·높이 허용 범위에 들어간 상태 |
| Input | Low-dimensional state, RGB/depth 사용하지 않음 |

### 데이터 생성 과정

```text
Teleoperation (Keyboard / SpaceMouse)
    ↓
Raw demonstrations (HDF5)
    ↓
Isaac Lab Mimic subtask annotation
    ↓
Isaac Lab Mimic demonstration generation
    ↓
Robomimic BC-RNN training
    ↓
Toy2Box rollout evaluation
```

실험에는 Base generated와 Choi generated 데이터셋을 사용했다.

| 데이터셋 | Demonstrations | 전체 Steps |
|---|---:|---:|
| Base generated | 1,000 | 606,172 |
| Choi v2 generated | 1,000 | 577,803 |

Choi v1 데이터셋도 일부 모델 학습에 사용했다. 데이터셋 버전이 다른 실험은 관측 구성뿐 아니라 학습 데이터도 달라진다.

## 3. Observation

### 3.1 Base — 23-D

| Key | 차원 | 내용 |
|---|---:|---|
| `eef_pos` | 3 | End-Effector 위치 |
| `eef_quat` | 4 | End-Effector 방향 (quaternion, wxyz) |
| `gripper_pos` | 2 | Gripper 관절 위치 |
| `toy_truck_positions` | 3 | ToyTruck 위치 |
| `toy_truck_orientations` | 4 | ToyTruck 방향 |
| `box_positions` | 3 | Box 위치 |
| `box_orientations` | 4 | Box 방향 |
| **Total** | **23** | |

### 3.2 Relative Position — +3-D

EEF에서 ToyTruck 원점까지의 상대 위치 벡터를 추가한다.

```python
eef_to_toy = toy_truck_positions - eef_pos
```

두 위치는 동일한 좌표계 기준이며, 상대 위치는 기존 관측으로부터 계산한다. 새로운 센서 정보가 아니라 물체까지의 방향과 거리 관계를 입력에 직접 표현한 값이다.

```text
23-D Base + 3-D Relative Position = 26-D
```

### 3.3 Joint State — +68-D

`joint_pos`와 `joint_vel`을 각각 34차원씩 추가한다.

```text
23-D Base + 34-D Joint Position + 34-D Joint Velocity = 91-D
```

관절 위치와 속도는 기준 관절 상태에 대한 상대값을 사용한다.

> 실험 재현 시 [BC-RNN 설정 파일](source/agibot_toy2box/agents/robomimic/bc_rnn_low_dim.json)의 관측 키를 확인해야 한다. 현재 설정 파일은 위의 23-D 기본 구성과 다르므로, 학습하려는 모델에 맞춰 키를 지정한다.

## 4. BC-RNN 학습 설정

Robomimic의 **Behavior Cloning + LSTM + GMM** 정책을 사용했다. LSTM은 이전 관측의 정보를 hidden state로 유지하고, GMM은 다음 행동의 확률분포를 표현한다.

| 항목 | 설정 |
|---|---|
| Algorithm | BC-RNN |
| Recurrent network | LSTM, 2 layers |
| Hidden dimension | 400 |
| RNN horizon | 10 |
| Sequence length | 10 |
| Frame stack | 1 |
| Open-loop | `False` |
| Action distribution | GMM, 5 modes |
| Loss | Negative log-likelihood |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Batch size | 100 |
| Updates per epoch | 100 |
| Training epochs | 2,000 |
| Nominal gradient updates | 200,000 |
| Default training seed | 101 |
| Observation normalization | `False` |
| Training-time validation | 비활성화 |

`Sequence length=10`은 연속 10개 시점의 관측과 행동을 이용해 학습한다는 뜻이다. 학습은 epoch당 100번의 배치 업데이트로 구성되므로, 2,000 epoch는 설정상 총 200,000번의 업데이트에 해당한다.

## 5. 평가 결과

### 5.1 모델별 성공률

| Obs | 모델 | 성공 / 전체 | 성공률 |
|---:|---|---:|---:|
| **23-D** | **`bc_choi_gen_23`** | **26 / 50** | **52%** |
| 23-D | `bc_choi_gen` | 15 / 50 | 30% |
| **26-D** | **`bc_base_gen_26`** | **49 / 100** | **49%** |
| **91-D** | **`bc_choi_gen_91`** | **20 / 100** | **20%** |
| 91-D | `bc_base_gen` | 9 / 50 | 18% |
| 91-D | `bc_choi_gen_v2` | 4 / 50 | 8% |

### 5.2 차원별 최고 성공률

각 관측 구성에서 가장 높은 성공률을 기록한 모델을 비교했다.

```text
23-D  ██████████████████████████  52% (26/50)
26-D  ████████████████████████    49% (49/100)
91-D  ██████████                  20% (20/100)
```

23-D 모델이 52%로 가장 높았고, 상대 위치 3차원을 추가한 26-D 모델은 49%를 기록했다. 관절 정보를 포함한 91-D 모델의 최고 성공률은 20%였다.

### 5.3 Episode Length

| 모델 | Evaluation seed | 성공률 | Episode length (mean ± std) |
|---|---:|---:|---:|
| `bc_choi_gen_91` | 42 | 20% (20/100) | 762.69 ± 80.11 steps |
| `bc_choi_gen_101` | 42 | 36% (36/100) | 715.67 ± 115.99 steps |

`bc_choi_gen_101`은 Observation 차원이 정리되지 않아 차원별 성공률 비교에서는 제외했다. 위 두 평가는 각각 seed 42에서 100회 수행한 결과다.

## 6. 실행 방법

### 설치

Isaac Lab v2.3.2와 호환 Isaac Sim 환경을 사용한다. 로봇과 작업 환경의 USD asset은 Nucleus에서 불러온다.

```bash
conda activate isaaclab
source env.sh
bash setup.sh
```

Robomimic이 설치되지 않은 경우 Isaac Lab 루트에서 `./isaaclab.sh -i robomimic`을 실행한다.

### 데이터 수집 및 증강

```text
datasets/toy2box_raw.hdf5
    → datasets/toy2box_annotated.hdf5
    → datasets/toy2box_generated.hdf5
```

Teleoperation, annotation, Mimic 생성에 사용하는 전체 명령은 [command.txt](command.txt)에 정리되어 있다.

### BC-RNN 학습

학습 전에 [BC-RNN 설정 파일](source/agibot_toy2box/agents/robomimic/bc_rnn_low_dim.json)의 Observation 키를 실험하려는 입력 차원에 맞춘다.

```bash
python scripts/robomimic/train.py \
  --task Agibot-Toy2Box-RmpFlow-v0 \
  --algo bc \
  --dataset ./datasets/toy2box_generated.hdf5 \
  --epochs 2000 \
  --seed 101 \
  --norm False \
  --output_dir ./output/bc_run
```

학습 로그와 체크포인트는 `output/bc_run/` 아래에 저장된다.

### Rollout 평가

```bash
python scripts/robomimic/play.py \
  --task Agibot-Toy2Box-RmpFlow-v0 \
  --checkpoint ./output/bc_run/model_epoch_2000.pth \
  --seed 42 \
  --num_rollouts 100 \
  --horizon 800 \
  --headless
```

`--seeds 42 43 44`로 여러 평가 seed를 사용할 수 있으며, `--video --video_folder ./videos/bc_run`을 추가하면 각 rollout 영상을 저장할 수 있다.

## 7. 결과 분석

이번 실험에서는 **기본 상태만 사용하는 23-D 모델이 52%로 가장 높은 성공률**을 보였다. 상대 위치를 명시적으로 추가한 26-D 모델도 49%로 비슷한 수준의 성능을 기록했다. 반면 전체 관절 상태를 추가한 91-D 모델은 최고 20%였다.

이는 Toy2Box에서 관측 차원을 늘리는 것만으로 성공률이 높아지지는 않는다는 점을 보여준다. 특히 EEF와 물체의 Pose만으로도 작업을 학습할 수 있으며, 상대 위치는 해당 관계를 직접 표현하는 보조 정보로 활용할 수 있다.

다만 실험마다 사용한 데이터셋과 평가 횟수가 일부 다르므로 **성능 차이를 관측 차원만의 효과로 해석하기는 어렵다.** 상대 위치와 관절 정보의 영향을 더 명확히 비교하려면 동일한 데이터셋과 학습·평가 seed를 사용한 추가 실험이 필요하다.

---

프로젝트 구조와 환경 설정은 [project.md](project.md), 상세 명령은 [command.txt](command.txt), 원본 코드 출처는 [UPSTREAM.md](UPSTREAM.md)에서 확인할 수 있다.
