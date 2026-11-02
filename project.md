# Agibot Toy2Box imitation-learning 프로젝트

최종 정리: 2026-10-20

## 목표

Isaac Lab의 Agibot A2D 오른팔로 장난감 트럭을 집어 상자에 넣는 `Toy2Box` task를 사용한다.
사람의 teleoperation 시연을 HDF5로 수집하고, Isaac Lab Mimic으로 시연을 증강한 뒤 robomimic의
recurrent BC policy를 학습·평가하는 데 필요한 코드를 이 저장소 안에 모았다.

## 구조

```text
robomimic/
├── command.txt                         # 수집부터 평가까지 실행 명령
├── env.sh                             # Isaac Sim standalone 환경 변수 설정
├── project.md                         # 프로젝트 구조와 동작 설명
├── setup.sh                           # Isaac Lab 환경 검사 및 editable install
├── source/
│   ├── agibot_toy2box/
│   │   ├── __init__.py                # base/mimic Gym task 등록
│   │   ├── agibot.py                  # Agibot A2D articulation/actuator 설정
│   │   ├── place_toy2box_env_cfg.py   # scene, action, observation, termination
│   │   ├── mimic_env_cfg.py           # Mimic subtask 및 generation 설정
│   │   └── agents/robomimic/
│   │       └── bc_rnn_low_dim.json    # Toy2Box용 recurrent BC 설정
│   └── isaaclab_mimic/                # Isaac Lab Mimic v2.3.2 Python source 전체
├── scripts/
│   ├── teleoperation/                 # 조작, 시연 기록, replay
│   ├── isaaclab_mimic/                # annotation 및 데이터 증강
│   └── robomimic/                     # BC 학습, 재생, robust evaluation
├── datasets/                          # 생성한 HDF5 저장 위치
├── logs/                              # robomimic 학습 결과
└── docs/                              # upstream imitation-learning 문서
```

## 등록된 환경

- `Agibot-Toy2Box-RmpFlow-v0`: 시연 수집, replay, BC 학습 및 평가용 환경
- `Agibot-Toy2Box-RmpFlow-Mimic-v0`: subtask annotation과 Mimic 데이터 생성용 환경

기존 Isaac Lab task ID와 충돌하지 않도록 별도의 ID를 사용한다.

## Task 구성

- Robot: Agibot A2D
- Control: 오른팔 7-DoF RMPFlow relative pose action + binary parallel gripper action
- Objects: `ToyTruck`과 `Box`
- Reset randomization: 트럭과 상자의 XY 위치 및 yaw
- Success: 트럭이 상자의 XY/높이 threshold 안에 들어간 상태
- Subtasks: `toy_truck` grasp, 이후 `box` placement
- Teleoperation devices: keyboard, SpaceMouse
- Recording viewports: D405 Eye + Box 상공 Top + 몸체 중앙 Front를 항상 표시. `--cameras`는 RGB 기록 대상만 선택
- Asset source: Isaac Lab Nucleus의 Agibot, ToyTruck, Box 및 SeattleLabTable USD

USD 파일은 저장소에 복제하지 않았다. 실행 시 `ISAACLAB_NUCLEUS_DIR`와 `ISAAC_NUCLEUS_DIR`에서
읽으므로 Nucleus asset 접근이 필요하다.

## 데이터 흐름

```text
teleoperation
  -> datasets/toy2box_raw.hdf5
  -> subtask annotation
  -> datasets/toy2box_annotated.hdf5
  -> Isaac Lab Mimic generation
  -> datasets/toy2box_generated.hdf5
  -> robomimic recurrent BC
  -> logs/robomimic/...
```

`annotate_demos.py`와 `generate_dataset.py`에는 Mimic task ID를 명시한다. BC 학습과 policy 재생에는
base task ID를 사용한다. 상세 명령은 `command.txt`에 있다.

BC-RNN과 ACT의 새 학습은 `bc_choi_gen` 및 IsaacLab_RS BC 설정과 같은 23차원 관측을 사용한다.
관측 키는 `eef_pos`, `eef_quat`, `gripper_pos`, `toy_truck_positions`, `toy_truck_orientations`,
`box_positions`, `box_orientations`이며, 관절 위치·속도와 `eef_to_toy`는 기본 입력에서 제외한다.
두 학습 스크립트 모두 `--norm True / False`로 관측 정규화를 선택하며 기본은 False다.
True이면 학습 데이터의 관측별 mean/std(`std + 1e-3`)를 사용하고, 통계를 체크포인트에 저장하여
play에서도 동일하게 적용한다. 시퀀스 샘플링과 패딩은 robomimic SequenceDataset을 사용한다.
기존 91·94차원 모델은 저장된 구성을 유지하므로, 23차원 모델은 resume 없이 새로 학습해야 한다.
재개 중 정규화 옵션을 바꾸는 것은 허용하지 않는다.

상대 위치 관측 코드는 유지한다. BC JSON에 `eef_to_toy`를 추가하면
`toy_truck_positions - eef_pos`를 기존 HDF5에서 계산하며 play에서도 같은 계산을 사용한다.

BC 평가에서는 `scripts/robomimic/play.py --seed 101`로 평가 seed를 지정한다(기본 101).
`--seeds 101 102 103 --num_rollouts 50`은 seed별로 환경을 새로 생성해 각각 50회, 총 150회 평가한다.
환경 생성 전부터 seed를 적용하며, 결과에는 seed별 성공률과 seed 간 성공률 평균·표준편차(`ddof=0`),
전체 rollout의 에피소드 길이 평균·표준편차가 출력된다. seed가 하나일 때 seed 간 표준편차는 0이다.

BC 학습 결과는 ACT처럼 `--output_dir` 바로 아래에 `model_epoch_XXXX.pth`, `config.json`,
`log.txt`, TensorBoard `tb/`로 저장한다. 실험명·timestamp·models·logs 중간 폴더는 만들지 않는다.
`--name`은 로그에 표시하는 실험 이름으로만 사용한다.

RNN 학습은 `--resume /PATH/TO/model_epoch_1000.pth --epochs 2000`으로 1001~2000 epoch를 이어서
수행할 수 있다. `--epochs`는 최종 epoch이며, 모델 구조와 관측 키는 체크포인트 설정을 유지한다.
기존 모델 전용 체크포인트는 파일명의 `epoch_<숫자>`에서 시작 epoch를 읽고 옵티마이저와 LR scheduler는
새로 시작한다. 새 체크포인트에는 epoch, 옵티마이저, LR scheduler, RNG, 최적 validation loss와
관측 정규화 통계도 저장한다. 재개 시 DataLoader는 새로 만들며, `--output_dir`을 생략하면 체크포인트가
있는 폴더를 그대로 사용한다. `config.json`과 같은 이름의 모델은 덮어쓰고, 텍스트 로그와 TensorBoard
기록은 이어 쓴다. `--output_dir`을 지정하면 해당 폴더 바로 아래에 저장한다. 예전 models/logs 구조의
체크포인트는 `--output_dir`을 생략하면 기존 모델·로그 위치를 유지한다.
데이터를 옮겼으면 `--dataset`으로 경로를 지정하고, action 정규화를 사용했다면 기존과 같은 정규화된
데이터셋을 사용한다. 재개 시 `--normalize_training_actions`로 다시 변환하지 않는다.

`record_demos.py`는 기본적으로 Eye, Top, Front camera view를 조작 화면에만 표시한다. RGB를 HDF5 observation으로
저장하려면 `--record_camera_data`를 명시하며, 기본값은 비활성화다.

ACT 학습도 지원한다. `scripts/act/train.py --enable_cam False`는 상태 관측만,
`--enable_cam True --cameras eye top front`는 지정한 최대 3개의 RGB 카메라를 사용한다.
BC-RNN과 같은 robomimic SequenceDataset 전처리로 action chunk를 학습한다(CLI 기본 32스텝).
설명과 추론 API는 [docs/act.md](docs/act.md), 실행 명령은 `command.txt`의 TRAIN ACT 절에 있다.

## 설치 조건

- Isaac Lab v2.3.2 및 호환 Isaac Sim
- Isaac Lab을 실행할 수 있는 conda 환경
- 데이터 증강용 `isaaclab_mimic` 의존성
- BC 학습용 `robomimic` v0.4.0

먼저 Isaac Lab 환경을 활성화하고 `source env.sh`, `bash setup.sh` 순서로 실행한다. 새 터미널을 열 때마다
`source env.sh`를 다시 실행해야 한다. robomimic이 없다면 Isaac Lab 루트에서
`./isaaclab.sh -i robomimic`으로 설치한다.

## Upstream 보존 범위

복사 기준과 원본 경로는 `UPSTREAM.md`에 기록했다. `source/isaaclab_mimic`은 알고리즘을 추적하고
수정할 수 있도록 전체 Python source를 가져왔고, task package와 실행 script에는 로컬 환경 등록 import만
추가했다.
