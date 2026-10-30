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

기존 `bc_rnn_low_dim.json`의 low_dim 키와 순서를 그대로 사용한다.
`joint_pos`, `joint_vel`, 물체/상자 pose, EEF pose, gripper 상태를 연결한다.
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
`--hdf5_normalize_obs True`이면 robomimic SequenceDataset이 계산한 관측별 mean/std로 정규화한다.
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

```bash
# RNN과 같은 정규화 기본값(현재 비활성) + temporal ensemble 기본 활성
python scripts/act/train.py \
    --dataset datasets/toy2box_raw.hdf5 \
    --steps 100000 \
    --output_dir output/act_ensemble

# RNN에서 지원하는 관측/action 정규화 옵션도 함께 사용
python scripts/act/train.py \
    --dataset datasets/toy2box_raw.hdf5 \
    --hdf5_normalize_obs True \
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
기존 epoch 체크포인트도 재생할 수 있다.
중간 중단 시 CSV에는 마지막 로그 시점까지 남고, 모델은 마지막 체크포인트 시점까지 남는다.
기존 `metrics.csv`가 있는 출력 폴더는 덮어쓰지 않으므로 새 학습에는 새 `--output_dir`를 사용한다.
자동 검증 분할이나 rollout 평가는 이 학습 스크립트에 포함하지 않았다.
