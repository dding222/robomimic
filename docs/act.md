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
백본은 기본 ResNet18이고 사전학습 가중치는 사용하지 않아 자동 다운로드가 없다.

## Sequence preprocessing

기존 `bc_rnn_low_dim.json`의 low_dim 키와 순서를 그대로 사용한다.
`joint_pos`, `joint_vel`, 물체/상자 pose, EEF pose, gripper 상태를 연결한다.
로봇 상태만으로도 ACT가 실행되며 별도 environment_state 입력은 필요 없다.

BC-RNN과 같은 `robomimic.utils.dataset.SequenceDataset`을 사용한다.
기본 `seq_length=10`, `frame_stack=1`, 에피소드 경계에서 마지막 값 반복 패딩,
관측 정규화 비활성화를 적용한다. 서로 다른 에피소드는 연결하지 않는다.
전체 영상을 메모리에 캐시하지 않도록 low_dim 캐시를 사용한다.

ACT에는 구간 첫 관측을 입력하고 같은 시점부터의 action 10개를 목표로 준다.
패딩 마스크는 ACT loss에 맞춰 반전하며 패딩된 action은 L1 loss에서 제외한다.
이 방식은 RNN의 시퀀스 데이터 전처리를 재사용하는 것이며 모델 자체는 LSTM이 아닌 ACT다.

## Actions and inference

학습 타깃은 HDF5 `data/demo_*/actions`의 원래 7차원 값이다.
순서는 `[dx, dy, dz, rx, ry, rz, gripper]`이며 RMPFlow relative pose 6개와 binary gripper 명령이다.
action 정규화, tanh 또는 clipping을 추가하지 않는다. 이미 정규화된 HDF5를 넣으면 그 값으로 학습되므로
원래 환경 action 단위로 출력하려면 원본 스케일의 HDF5를 사용해야 한다.

학습 `forward()`는 loss와 metrics를 반환한다. 추론 `predict_action_chunk()`는 `(B, K, 7)`,
`select_action()`은 `(B, 7)`을 반환한다. 기본적으로 K개를 큐에 넣고 한 스텝씩 실행한다.
환경을 리셋할 때 `policy.reset()`을 호출해 action 큐를 비운다.
이 체크포인트는 별도 ACT 형식이므로 기존 `scripts/robomimic/play.py`로 로드하지 않는다.
ACT 재생은 전용 스크립트를 사용한다. 카메라와 action chunk 설정은 체크포인트에서 자동 복원한다.
GUI에서는 Eye, Top, Front 세 뷰포인트를 모두 표시하며 `--headless`로 화면 없이 평가할 수 있다.

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

기본 학습 길이는 1,000 epoch이며 `--epochs`로 변경한다.
매 epoch의 샘플 수 가중 평균 `loss`, `l1`, `kl`을 `metrics.csv`에 기록하고 즉시 flush한다.
100 epoch마다 `model_epoch_100.pt`, `model_epoch_200.pt`처럼 체크포인트를 보존한다.
마지막 epoch는 100의 배수가 아니어도 저장하며, 저장 시 `latest.pt`도 갱신한다.
체크포인트에는 모델, 구조 설정, 관측 순서, optimizer 상태와 해당 epoch의 지표가 들어간다.
중간 중단 시 CSV에는 완료된 epoch까지 남고, 모델은 마지막 체크포인트 시점까지 남는다.
기존 `metrics.csv`가 있는 출력 폴더는 덮어쓰지 않으므로 새 학습에는 새 `--output_dir`를 사용한다.
자동 검증 분할이나 rollout 평가는 이 학습 스크립트에 포함하지 않았다.
