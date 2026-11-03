# Agibot Toy2Box — BC-RNN Observation Ablation

Isaac Lab v2.3.2의 **Agibot A2D Toy2Box** 환경에서 시연 데이터로 **BC-RNN (LSTM + GMM)** 정책을 학습하고, **Observation 구성 및 데이터셋에 따른 작업 성공률**을 비교한 실험이다.

이 README는 실험 비교를 중심으로 정리한다. 전체 환경·코드 구조는 [project.md](project.md), 단계별 실행 명령은 [command.txt](command.txt), ACT 관련 내용은 [docs/act.md](docs/act.md)를 참고한다.

> **결과 출처 및 범위:** 아래 성공률은 별도로 기록된 모델별 rollout 평가 결과이다. 모델마다 평가 횟수(50 또는 100회), 데이터셋, 평가 seed 등이 완전히 같다고 확인되지 않았으므로, 단순한 성공률 순위가 엄밀한 ablation 결과를 의미하지는 않는다. 학습을 새로 실행해 검증한 수치는 아니다.

---

## 1. 실험 구성

| Observation | 차원 | 구성 | 비교 목적 |
|---|---:|---|---|
| **Base State** | **23-D** | EEF Pose + Gripper + Toy Pose + Box Pose | 상태 관측만 이용한 기본 모델 |
| **Base + Relative Position** | **26-D** | 23-D + EEF→Toy 상대 위치 XYZ | 명시적 상대 위치 정보의 효과 |
| **Base + Joint State** | **91-D** | 23-D + Joint Position 34-D + Joint Velocity 34-D | 전체 관절 상태 추가 효과 |
| **Base + Joint + Relative Position** | **94-D** | 91-D + EEF→Toy 상대 위치 XYZ | 관절 상태가 있을 때 상대 위치 추가 효과 |

비교 기준은 **최종 Toy2Box 성공률(success rate)**이다. 일부 평가는 episode length도 기록했다.

---

## 2. 환경 및 데이터 생성

### Task

| 항목 | 구성 |
|---|---|
| Simulator | Isaac Lab v2.3.2 / Isaac Sim |
| Robot | Agibot A2D, 오른팔 |
| Base task | `Agibot-Toy2Box-RmpFlow-v0` |
| Mimic task | `Agibot-Toy2Box-RmpFlow-Mimic-v0` |
| Object | ToyTruck, Box |
| Control | RMPFlow 기반 오른팔 relative EE pose + gripper |
| Action | **7-D:** `[dx, dy, dz, rx, ry, rz, gripper]` |
| Randomization | 장난감·상자의 XY 위치와 yaw |
| Success | 장난감이 상자 내부의 설정된 위치·높이 조건을 만족 |
| Policy input | **Low-dimensional state only** (RGB / depth 미사용) |

### Demonstration → Policy

```text
Teleoperation (keyboard / SpaceMouse)
    → Raw demonstrations (HDF5)
    → Isaac Lab Mimic subtask annotation
    → Isaac Lab Mimic demonstration generation
    → Robomimic SequenceDataset
    → BC-RNN (LSTM + GMM)
    → Toy2Box rollout evaluation
```

- **Base generated:** 로컬 기록 기준 1,000 demonstrations, 총 606,172 steps.
- **Choi v2 generated:** 로컬 기록 기준 1,000 demonstrations, 총 577,803 steps.
- **Choi v1:** 과거 체크포인트에 `team11_choi_mimic_v1_500.hdf5` 경로가 기록되어 있으나, 원본 HDF5의 실제 demo 수·길이는 확인되지 않았다.

데이터셋 버전이 다른 모델의 결과를 Observation 차원만의 효과로 해석하지 않는다.

---

## 3. Observation

### 3.1 기본 상태 — 23-D

| Key | Dim | 의미 |
|---|---:|---|
| `eef_pos` | 3 | 로봇 기준 End-Effector 위치 |
| `eef_quat` | 4 | End-Effector 방향 quaternion (wxyz) |
| `gripper_pos` | 2 | 그리퍼 관절 상태 |
| `toy_truck_positions` | 3 | 장난감 위치 |
| `toy_truck_orientations` | 4 | 장난감 방향 |
| `box_positions` | 3 | 상자 위치 |
| `box_orientations` | 4 | 상자 방향 |
| **Total** | **23** | |

### 3.2 EEF→Toy 상대 위치 — +3-D

```python
eef_to_toy = toy_truck_positions - eef_pos
```

장난감 원점까지의 **방향을 포함한 3차원 위치 차이**이며, 두 위치는 같은 좌표계 기준이다. 기존 23-D 상태만으로도 계산할 수 있으므로 새로운 센서 측정값은 아니다.

- `23 + 3 = 26-D`
- `91 + 3 = 94-D`

### 3.3 관절 상태 — +68-D

- `joint_pos`: 34-D, 기준 관절 위치에 대한 상대 관절 위치
- `joint_vel`: 34-D, 기준 관절 속도에 대한 상대 관절 속도
- `23 + 34 + 34 = 91-D`

> **재현 시 주의:** 현재 `main` 브랜치의 [`bc_rnn_low_dim.json`](source/agibot_toy2box/agents/robomimic/bc_rnn_low_dim.json)에는 `box_orientations` 대신 `eef_to_toy`가 포함되어 있어, 나열된 관측 차원 합계가 **22-D**이다. 이는 이 README에서 정리한 과거 **23/26/91/94-D 실험 구성과 다르다.** 기존 결과는 현재 JSON이 아니라 당시 체크포인트의 Observation 설정을 기준으로 해석해야 한다. 새 학습에서 23-D를 재현하려면 3.1의 7개 키를 사용하도록 JSON을 먼저 맞춘다. 코드/설정 파일은 이 문서 작성 과정에서 변경하지 않았다.

---

## 4. BC-RNN 학습 설정

[Robomimic BC 설정](source/agibot_toy2box/agents/robomimic/bc_rnn_low_dim.json) 및 기존 체크포인트 분석 기록 기준.

| 항목 | 설정 |
|---|---|
| Algorithm | Behavior Cloning (BC-RNN) |
| Recurrent model | LSTM, 2 layers, hidden size **400** |
| RNN horizon | 10 |
| Sequence length | 10 |
| Frame stack | 1 |
| Open-loop | `False` (매 step 현재 관측 반영) |
| Action distribution | **GMM**, 5 modes |
| Loss | Action negative log-likelihood |
| Optimizer | Adam |
| Initial learning rate | `1e-3` |
| Batch size | 100 |
| Updates / epoch | 100 |
| Epoch configuration | 2,000 |
| Nominal updates | **200,000** (2,000 × 100) |
| Default train seed | 101 |
| HDF5 observation normalization | `False` |
| Training-time validation / rollout | 비활성화 |

**Sequence length 10**은 연속된 관측·행동을 RNN 학습에 제공하는 길이이며, **RNN horizon 10**은 hidden state를 재설정하는 구간과 관련된다. ACT의 action chunk 크기와는 다른 개념이다.

Loss는 GMM의 음의 로그우도이므로 음수가 나올 수 있으며, 학습 Loss가 낮다고 실제 rollout 성공률이 높다고 단정할 수 없다. 일부 기존 체크포인트는 재개 시 Adam 상태가 초기화된 이력이 있으므로 학습 조건이 완전히 통제된 것은 아니다.

---

## 5. Rollout 평가 결과

### 5.1 모델별 결과

성공률은 각각 `성공 횟수 / 평가 횟수`로 계산했다.

| Obs | 모델 / 평가 기록 | 성공 / 전체 | 성공률 | 비고 |
|---:|---|---:|---:|---|
| **23-D** | **`bc_choi_gen_23`** | **26 / 50** | **52%** | 신규 기록; 세부 데이터 버전·seed 미확인 |
| 23-D | `bc_choi_gen` (기존 평가) | 15 / 50 | 30% | 체크포인트 기록상 Choi v1 |
| **26-D** | **`bc_base_gen_26`** | **49 / 100** | **49%** | Base 계열; 평가 seed 미확인 |
| **91-D** | **`bc_choi_gen_91`** | **20 / 100** | **20%** | Train seed 101 / Eval seed 42 |
| 91-D | `bc_base_gen` | 9 / 50 | 18% | Base generated |
| 91-D | `bc_choi_gen_v2` | 4 / 50 | 8% | Choi v2 generated |
| **94-D** | **`bc_choi_gen_pose_err`** | **10 / 50** | **20%** | Choi v2; 재개 시 Adam 상태 초기화 이력 |
| 94-D | `bc_base_gen_pose_err` | 7 / 50 | 14% | Base generated |

**별도 기록:** `bc_choi_gen_101`은 100회 중 36회 성공(**36%**, 평가 seed 42)이지만 **Observation 차원이 확인되지 않아** 차원별 표와 순위에는 포함하지 않았다. 모델명의 접미사만으로 차원을 추정하지 않는다.

### 5.2 Observation 차원별 최고 성공률

아래는 차원별로 기록된 **최고값**만 비교한 것이며, 동일 데이터·동일 seed를 통제한 실험이 아니다.

```text
23-D  ██████████████████████████  52% (26/50)
26-D  ████████████████████████    49% (49/100)
91-D  ██████████                  20% (20/100)
94-D  ██████████                  20% (10/50)
```

현재 기록상 최고 성공률은 **23-D 52%**, 다음은 **26-D 49%**다. 다만 3%p 차이만으로 23-D가 일반적으로 우수하다고 결론 내릴 수 없다.

### 5.3 Episode length가 기록된 평가

| 모델 | Eval seed | 성공률 | Episode length (mean ± std, steps) |
|---|---:|---:|---:|
| `bc_choi_gen_91` | 42 | 20/100 (20%) | 762.69 ± 80.11 |
| `bc_choi_gen_101` (차원 미확인) | 42 | 36/100 (36%) | 715.67 ± 115.99 |

각각 단일 평가 seed의 100회 rollout 결과다. Seed 1개에서 seed 간 표준편차 0이 출력되더라도, 그것은 여러 평가 seed에 대한 성능 안정성을 의미하지 않는다.

---

## 6. 결과 해석과 한계

- **저차원 상태가 높은 성공률을 기록:** 현재 최고 성능은 23-D 52%, 26-D 49%. 관절 전체 정보를 입력하지 않아도 Toy2Box 동작을 학습할 수 있음을 보여준다.
- **상대 위치 관측의 효과는 확정 불가:** 23-D와 26-D의 최고 모델은 학습 데이터·평가 조건이 통제되지 않았다. Base generated의 기존 91-D→94-D 비교는 **18%→14%**였고, Choi v2의 91-D→94-D 비교는 **8%→20%**였으나 후자는 optimizer 재개 이력도 다르다.
- **관측 차원과 성공률 사이 인과관계 미확정:** 23-D, 26-D, 91-D, 94-D 실험은 일부 데이터셋, 학습 및 평가 이력이 다르다.
- **동일 이름 계열의 평가를 구분:** `bc_choi_gen`의 30%와 `bc_choi_gen_23`의 52%는 동일 체크포인트 재평가인지 확인되지 않아 별도 기록한다.
- **추가 평가 필요:** 같은 데이터셋, 학습 seed, 평가 seed 및 episode horizon을 고정한 반복 실험이 필요하다.

---

## 7. 실행

### 7.1 설치

Isaac Lab v2.3.2와 호환 Isaac Sim 환경이 필요하다. Robot·table·toy 등의 USD asset을 불러오기 위한 Nucleus 접근도 필요하다.

```bash
conda activate isaaclab
source env.sh
bash setup.sh
```

Robomimic이 설치되지 않았다면 Isaac Lab 루트에서 `./isaaclab.sh -i robomimic`을 실행한다.

### 7.2 Demonstration 수집 및 증강

정확한 teleoperation, annotation, Mimic generation 명령은 [command.txt](command.txt)에 정리되어 있다.

```text
datasets/toy2box_raw.hdf5
    → datasets/toy2box_annotated.hdf5
    → datasets/toy2box_generated.hdf5
```

### 7.3 BC-RNN 학습

**먼저** `bc_rnn_low_dim.json`의 `observation.modalities.obs.low_dim`을 원하는 차원 구성으로 맞춘다. 새 학습의 기본 관측은 현재 파일 내용에 의해 결정된다.

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

학습 결과는 `output/bc_run/model_epoch_XXXX.pth`, `config.json`, `log.txt`, `tb/`에 저장된다.

### 7.4 평가

```bash
python scripts/robomimic/play.py \
  --task Agibot-Toy2Box-RmpFlow-v0 \
  --checkpoint ./output/bc_run/model_epoch_2000.pth \
  --seed 42 \
  --num_rollouts 100 \
  --horizon 800 \
  --headless
```

- `--seed`는 평가 seed이며 학습 seed와 별개다.
- `--seeds 42 43 44`를 사용하면 seed별 결과와 seed 간 평균·표준편차를 출력한다.
- 로더는 체크포인트에 저장된 Observation 키와 정규화 설정을 복원한다.
- 영상은 `--video --video_folder ./videos/bc_run`을 추가해 rollout별로 저장할 수 있다.

현재 저장소에는 [`output/bc_choi_gen_26/model_epoch_2000.pth`](output/bc_choi_gen_26/model_epoch_2000.pth) 체크포인트가 있다. 이 파일은 **`bc_base_gen_26`의 49% 평가와 동일한 체크포인트라고 확인된 것은 아니므로**, 수치와 혼동하지 않는다.

---

## 8. 관련 문서

- [project.md](project.md) — 프로젝트 구조, 환경, 시연 데이터 흐름
- [command.txt](command.txt) — 수집, annotation, generation, 학습, 재생 명령
- [docs/act.md](docs/act.md) — ACT 학습 및 Temporal Ensembling
- [UPSTREAM.md](UPSTREAM.md) — Isaac Lab / Isaac Lab Mimic 원본 보존 내역

**ACT와 BC-RNN의 직접적인 정량 비교 결과는 이 문서에 포함하지 않는다.** 현재 표는 BC-RNN 평가 기록만 정리한 것이다.
