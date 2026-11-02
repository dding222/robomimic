# ACT training

ACT는 로컬 LeRobot의 Transformer/CVAE/ResNet 코드를 독립 PyTorch 모듈로 가져왔다.
학습은 Isaac Sim을 실행하지 않고 robomimic HDF5를 직접 읽는다.
설치와 학습 명령은 `command.txt`의 TRAIN ACT 절에 있다.

## Camera option

`--enable_cam False`가 기본값이다. 이미지 관측을 읽지 않으며 torchvision도 import하지 않는다.
ResNet, 이미지 projection, 카메라 위치 임베딩은 생성되지 않는다.
`--enable_cam True --cameras eye top front`이면 지정한 RGB 관측 1~3개를 사용한다.
이름은 `eye` → `hand_eye_cam`, `top` → `top_cam`, `front` → `front_cam`으로 연결된다.
기존 `--camera_keys hand_eye_cam top_cam` 구문도 호환된다.
키를 생략하면 첫 데모의 RGB 키를 자동 탐색하고 모든 데모에서 존재 여부와 형상을 검사한다.
카메라 데이터는 `record_demos.py --record_camera_data`로 수집해야 한다.
수집 GUI는 `--cameras` 선택과 무관하게 Eye, Top, Front 세 뷰포인트를 항상 표시한다.
`--record_camera_data --cameras eye top`이면 Eye와 Top RGB만 HDF5에 저장한다.
`--cameras`의 기본 기록 대상은 세 카메라이고, `--record_camera_data`를 생략하면 RGB는 저장하지 않는다.
`--headless` 실행에는 GUI 뷰포인트가 없다. ACT 학습의 `--cameras`는 학습에 사용할 관측 선택이다.
Front는 몸체 중앙의 승강 링크에 부착되어 책상 위를 전방으로 내려다본다.
Eye는 D405의 90도 보정 전 광학 회전값에서 영상 방향을 180도 반전했다. Top도 직전 설정에서 영상 방향을 180도 반전했다.
Top 위치는 `(0.3, -0.05, 0.75)`이며 Box 쪽으로 이동한 상태에서 월드 -Z축을 바라본다.
카메라 위치/방향은 `source/agibot_toy2box/cameras.py`에서 화면과 수집에 공통으로 적용한다.
RGB는 uint8 HWC에서 robomimic의 처리 함수를 통해 float32 CHW [0, 1]로 변환한다.
백본은 원본 ACT와 같은 ResNet18 ImageNet 사전학습 가중치를 사용한다.
카메라를 켰을 때 가중치 캐시가 없으면 torchvision이 다운로드한다. 카메라를 끄면 백본을 생성하지 않는다.

## Sequence preprocessing

기본 관측은 성공한 `output/bc_choi_gen/model_epoch_2000.pth`의 low_dim 키와 같은 23차원이다.
순서는 `eef_pos`(3), `eef_quat`(4), `gripper_pos`(2), `toy_truck_positions`(3),
`toy_truck_orientations`(4), `box_positions`(3), `box_orientations`(4)이다.
`joint_pos`, `joint_vel`은 새 학습 입력에서 제외한다. 이 키 목록은 `toy2box_act/dataset.py`에 정의한다.
BC-RNN 기본 JSON도 같은 7개 키를 사용하며, 두 모델의 상태 입력은 23차원으로 맞춘다.
기존 91차원 체크포인트의 재생·재개는 저장된 `state_keys`와 차원을 그대로 복원한다.
23차원 모델은 입력 projection 크기가 달라지므로 새 출력 폴더에서 처음부터 학습한다.
로봇 상태만으로도 ACT가 실행되며 별도 environment_state 입력은 필요 없다.

BC-RNN과 같은 `robomimic.utils.dataset.SequenceDataset`을 사용한다.
학습 CLI의 기본 `--action_chunk 32`를 SequenceDataset의 `seq_length`로 전달한다.
`frame_stack=1`, 에피소드 경계에서 마지막 값 반복 패딩,
관측 정규화 기본값은 BC-RNN JSON의 `train.hdf5_normalize_obs`를 따른다(현재 False).
서로 다른 에피소드는 연결하지 않는다.
전체 영상을 메모리에 캐시하지 않도록 low_dim 캐시를 사용한다.

ACT에는 구간 첫 관측을 입력하고 같은 시점부터 `action_chunk`개의 action을 목표로 준다.
관측 로더만 현재 시점 1개를 읽도록 확장해 사용하지 않는 미래 상태/RGB를 준비하지 않는다.
action chunk, 에피소드 경계 및 반복 패딩은 SequenceDataset의 처리를 그대로 사용한다.
패딩 마스크는 ACT loss에 맞춰 반전한다. L1은 원본처럼 패딩 위치 오차를 0으로 만든 뒤
전체 `(B, K, action_dim)`에 대해 평균한다. 유효 action 개수로 다시 나누지 않는다.
이 방식은 RNN의 시퀀스 데이터 전처리를 재사용하는 것이며 모델 자체는 LSTM이 아닌 ACT다.

## Training execution

원본 LeRobot처럼 TF32 행렬 연산과 cuDNN benchmark를 활성화한다.
DataLoader worker에서 관측 연결과 RGB 전처리를 마친 뒤 `pin_memory`로 결과를 고정하고,
CUDA로 `non_blocking` 전송한다. action과 패딩 마스크도 같은 경로로 전달한다.
AdamW와 gradient clipping은 원본 설정을 사용한다.
`--use_amp False`가 원본처럼 기본값이며 `--use_amp True`로 autocast와 GradScaler를 켤 수 있다.
현재 CLI의 배치 기본값은 100, worker 기본값은 4이고 `--batch_size`, `--num_workers`로 지정한다.
원본처럼 배치별 업데이트 1회를 1 step으로 세고 총 step 수로 학습을 종료한다.
데이터를 모두 사용하면 다시 섞어 순회한다. 로그·저장 시점에 데이터 순회를 초기화하지 않는다.
시작 로그에 적용된 배치/worker/AMP 및 step·로그·저장 간격을 표시하고 `config.json`에도 학습 설정을 저장한다.

## Actions and inference

학습 타깃은 HDF5 `data/demo_*/actions`의 원래 7차원 값이다.
순서는 `[dx, dy, dz, rx, ry, rz, gripper]`이며 RMPFlow relative pose 6개와 binary gripper 명령이다.
정규화 옵션을 생략하면 BC-RNN 기본 설정처럼 원래 상태/action 값을 사용한다.
`--norm True`이면 robomimic SequenceDataset이 계산한 관측별 mean/std로 정규화한다.
BC-RNN과 ACT 모두 같은 CLI 옵션을 사용한다. `--norm False`는 정규화를 끄며 기본값이다.
std의 `1e-3` 보정과 RGB의 HWC→CHW, /255 처리도 RNN과 같다. ImageNet mean/std는 추가하지 않는다.
`--normalize_training_actions`를 지정하면 로컬 BC-RNN과 같은 데이터 전체의 단일 min/max로
`2 * (action - min) / (max - min) - 1`을 적용한다. 축별 min/max나 LeRobot의 mean/std action 정규화가 아니다.
CVAE에도 정규화된 action 타깃을 입력하며, 추론 결과는 원래 action 단위로 자동 복원한다.
통계는 체크포인트의 `normalization`과 모델 buffer에 저장한다. 원본 HDF5는 수정하지 않는다.
이미 정규화된 HDF5만 제공하면 수집 당시의 action 단위를 알 수 없으므로 원본 스케일의 HDF5를 사용한다.

학습 `forward()`는 loss와 metrics를 반환한다. 추론 `predict_action_chunk()`는 `(B, K, 7)`,
`select_action()`은 `(B, 7)`을 반환한다. 새 학습의 기본값은 temporal ensemble 활성화(계수 0.01)다.
매 스텝 현재 관측으로 K개 action을 다시 예측하고, 같은 시점에 대한 겹치는 예측을 가중 평균한다.
로컬 LeRobot ACT의 온라인 알고리즘을 사용하며 가중치는 오래된 예측부터 `exp(-coeff * i)`다.
양수 계수는 오래된 예측에 더 높은 가중치를 주고, 0은 균등 평균이다. 값이 0인 action도 정상적으로 포함한다.
Ensemble 사용 시 `n_action_steps=1`이어야 한다. `--temporal_ensemble_coeff None`이면 기존 action 큐를 사용하고
`--n_action_steps` 생략 시 K개를 실행한다. `policy.reset()`은 큐와 ensemble 누적 상태를 모두 지운다.
이 체크포인트는 별도 ACT 형식이므로 기존 `scripts/robomimic/play.py`로 로드하지 않는다.
ACT 재생은 전용 스크립트를 사용한다. 카메라와 action chunk 설정은 체크포인트에서 자동 복원한다.
기존 체크포인트의 `seq_length` 설정도 로드 시 `action_chunk`로 변환한다.
GUI에서는 Eye, Top, Front 세 뷰포인트를 모두 표시하며 `--headless`로 화면 없이 평가할 수 있다.
기존 체크포인트에 새 설정이 없으면 정규화와 ensemble을 자동으로 켜지 않아 이전 동작을 유지한다.
기존 모델에도 `play.py --temporal_ensemble_coeff 0.01`로 ensemble을 적용할 수 있다.
`--disable_temporal_ensemble`로 재생 시 ensemble을 끌 수 있다. 정규화 변경은 해당 통계로 새로 학습해야 한다.

`play.py --check_observations`는 `--num_rollouts`만큼 데이터셋의 서로 다른 데모 상태를 환경에 대입한 뒤
관측 키별 값과 체크포인트에 기록된 ACT 상태 입력을 비교한다. 기본값은 각 데모의 전체 시점 비교다.
진단 모드의 `--num_rollouts 20`은 숫자 순서의 `demo_0`부터 `demo_19`까지 20개 에피소드를 검사한다.
정책 추론이나 action 적용은 하지 않는다. 일반 play 모드에서는 기존처럼 정책 rollout 횟수로 사용한다.
시뮬레이터를 시작하므로 명령을 직접 실행해야 하며 RGB는 비교하지 않는다.
`--check_obs_step 100`은 각 선택 데모에서 인덱스 100의 관측 하나만 비교한다.
0부터 100까지의 구간을 비교한다는 의미가 아니다.
`--check_obs_all_steps`를 명시해도 각 선택 데모의 0부터 마지막 인덱스까지 모든 관측을 비교한다.
단일 시점 옵션과 전체 시점 옵션은 동시에 사용하지 않는다. 전체 옵션은 `--check_observations`도 자동으로 켠다.
각 에피소드마다 관측별 최대 오차, 불일치 시점 수와 최초 불일치 인덱스, MATCHED/MISMATCH 결과를 표시한다.
마지막에는 전체 에피소드와 관측 시점의 일치 개수를 요약한다. 같은 데모를 반복해서 검사하지 않는다.
데이터의 `obs[t]`는 action 전, `states[t]`는 action 후 기록이므로 t>0에서는 `states[t-1]`을 복원한다.

```bash
python scripts/act/play.py \
    --checkpoint output/act_choi_gen/model_step_200000.pt \
    --check_observations --check_obs_step 100 --headless
```

```bash
# 20개 데모 각각의 전체 시점 비교
python scripts/act/play.py \
    --checkpoint output/act_choi_gen/latest.pt \
    --device cpu --headless --num_rollouts 20 --check_observations
```

관측이 일치한 뒤에는 `check_actions.py`로 기록된 정답 action과 모델 예측을 비교한다.
시뮬레이터를 시작하지 않고 저장된 관측을 모델에 입력하므로 실제 rollout 성공률 검사는 아니다.
모델과 정규화는 체크포인트에서 복원하고, 각 데모에서 ensemble 상태를 새로 시작한다.

```bash
python scripts/act/check_actions.py \
    --checkpoint output/act_choi_gen/model_step_200000.pt \
    --num_demos 20 --device cuda
```

`first_action`은 현재 관측에서 예측한 첫 action, `temporal_ensemble`은 기록 관측 순서대로
같은 시점의 예측을 평균한 action, `last_chunk_action`은 chunk 끝 시점에 대한 유효 예측이다.
위치 delta 오차를 mm, 회전 벡터 성분 오차를 degree, 그리퍼 부호 정확도를 %로 표시한다.
그리퍼 전환 정확도는 정답의 열기/닫기 전환 전후 5스텝 구간만 따로 계산한다.
회전 벡터 오차는 SO(3) 회전 거리와 다르며, 위치 delta 오차도 실제 EEF 경로 오차를 뜻하지 않는다.
각 데모 결과와 전체 평균을 출력한다. 실제 정책 실행 중의 누적 오차는 별도의 rollout에서 확인해야 한다.

```bash
# RNN과 같은 정규화 기본값(현재 비활성) + temporal ensemble 기본 활성
python scripts/act/train.py \
    --dataset datasets/toy2box_raw.hdf5 \
    --steps 100000 \
    --output_dir output/act_ensemble

# RNN에서 지원하는 관측/action 정규화 옵션도 함께 사용
python scripts/act/train.py \
    --dataset datasets/toy2box_raw.hdf5 \
    --norm True \
    --normalize_training_actions \
    --temporal_ensemble_coeff 0.01 \
    --steps 100000 \
    --output_dir output/act_normalized_ensemble
```

```bash
python scripts/act/play.py \
    --checkpoint logs/act_state/latest.pt \
    --device cpu \
    --num_rollouts 10
```

```python
from toy2box_act import ACTPolicy
from toy2box_act.dataset import prepare_observations
from robomimic.utils import obs_utils as ObsUtils

policy, state_keys = ACTPolicy.from_checkpoint("logs/act_state/latest.pt", device="cpu")
camera_keys = policy.config.camera_keys
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": state_keys, "rgb": camera_keys}})
policy.reset()
# env_obs는 배치 축이 있는 IsaacLab obs_dict["policy"]
batch = prepare_observations(env_obs, state_keys, camera_keys, "cpu")
action = policy.select_action(batch)  # (num_envs, 7), env.step(action)에 전달
```

기본 학습 길이는 원본과 같은 100,000 step이며 `--steps`로 변경한다. 이전 `--epoch`/`--epochs`는 사용하지 않는다.
기본 `--log_freq 200`으로 최근 로그 구간의 배치별 평균 `loss`, `l1`, `kl`을 `metrics.csv`에 기록하고 즉시 flush한다.
CSV 열은 `step,loss,l1,kl,data_s,update_s`이다. 마지막 step도 기록하며, `--log_freq 0`이면 마지막 step에서만 기록한다.
원본처럼 배치를 기다린 시간 `data_s`와 학습 업데이트 구간의 시간 `update_s`를 초 단위 평균으로 표시한다.
이는 CPU 시계로 측정한 구간 시간이며 CUDA 커널 자체의 실행 시간을 측정하는 프로파일러는 아니다.
기본 `--save_freq 1000`으로 `model_step_1000.pt`, `model_step_2000.pt`처럼 체크포인트를 보존한다.
마지막 step은 저장 간격의 배수가 아니어도 저장하며, 저장 시 `latest.pt`도 갱신한다.
체크포인트에는 모델, 구조 설정, 관측 순서, optimizer/GradScaler 상태와 step 및 지표가 들어간다.
새로 저장하는 체크포인트에는 Python/NumPy/PyTorch/CUDA 난수 상태도 포함한다.
기존 epoch 체크포인트도 재생할 수 있다.
중간 중단 시 CSV에는 마지막 로그 시점까지 남고, 모델은 마지막 체크포인트 시점까지 남는다.
기존 `metrics.csv`가 있는 출력 폴더는 덮어쓰지 않으므로 새 학습에는 새 `--output_dir`를 사용한다.
자동 검증 분할이나 rollout 평가는 이 학습 스크립트에 포함하지 않았다.

## Resume training

저장이 완료된 step 체크포인트에서 `--resume`으로 이어서 학습한다.
예를 들어 60,000 step 체크포인트는 60,001부터 시작하며 `--steps 100000`이면 40,000번 더 업데이트한다.
`--steps`를 생략하면 체크포인트의 원래 총 목표 step을 사용한다.

```bash
python scripts/act/train.py \
    --resume output/act_base_gen_steps/model_step_60000.pt \
    --steps 100000
```

데이터 경로, 모델/카메라/정규화 설정, 배치/worker/학습률/seed/AMP/로그·저장 간격을 체크포인트에서 복원한다.
관측 키와 차원을 검사하고 모델·optimizer·GradScaler를 불러온다. 정규화 통계도 저장된 값을 사용한다.
재개 시 `--norm`을 명시하면 체크포인트와 같은 값이어야 한다.
정규화 설정을 바꾸려면 resume 없이 새로 학습한다.
기존 기록을 보존하기 위해 기본 출력 폴더는 체크포인트 폴더 아래 `resume_step_60000`이다.
다른 새 폴더를 쓰려면 `--output_dir`를 지정한다. 재개 CSV에는 재개 이후의 기록만 저장한다.
이전 체크포인트에 난수 상태가 없더라도 재개할 수 있다. 난수 상태가 있으면 함께 복원한다.
데이터 순회는 다시 시작하므로 중단 없이 학습한 경우와 배치 순서가 완전히 같다는 의미는 아니다.
이전 epoch 체크포인트는 재생만 지원하며 `--resume`에는 step과 optimizer가 저장된 파일을 사용한다.
