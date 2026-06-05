#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   ./run.sh
#   DATA=KdV-PINN INPUT_PATH=data ./run.sh
#   PDE_RANKS="2:8,3:8,4:4" ./run.sh
#   PDE_PRUNE_METRIC=stridge PDE_STLSQ_THRESHOLD=0.03 PDE_STRIDGE_LAMBDA=1e-5 ./run.sh
#   TRUE_PDE_JSON='{"u*u_x": -1.0, "u_xx": 0.1}' ./run.sh  # override auto value
#   FORCE_RETRAIN=1 DISCOVER_PDE=0 EPOCHS=1000 ./run.sh


PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA="${DATA:-KS}"
INPUT_PATH="${INPUT_PATH:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
SEED="${SEED:-2026}"

NOISE_STD="${NOISE_STD:-0.0}"
TEST_SPLIT="${TEST_SPLIT:-0.2}"

NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-5}"
NEURONS_PER_LAYER="${NEURONS_PER_LAYER:-50}"
ACTIVATION="${ACTIVATION:-Rat}"

EPOCHS="${EPOCHS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LAMBDA_REG="${LAMBDA_REG:-0.0}"

DISCOVER_PDE="${DISCOVER_PDE:-1}"
PDE_MAX_ORDER="${PDE_MAX_ORDER:-3}"
PDE_RANK="${PDE_RANK:-4}"
PDE_RANKS="${PDE_RANKS:-2:8,3:8}"
PDE_DENSE_EPOCHS="${PDE_DENSE_EPOCHS:-0}"
PDE_SPARSE_EPOCHS="${PDE_SPARSE_EPOCHS:-2000}"
PDE_LEARNING_RATE="${PDE_LEARNING_RATE:-1e-3}"
PDE_LAMBDA_RIDGE="${PDE_LAMBDA_RIDGE:-1e-6}"
PDE_LAMBDA_G="${PDE_LAMBDA_G:-0.05}"
PDE_LAMBDA_ALPHA="${PDE_LAMBDA_ALPHA:-1e-4}"
PDE_LAMBDA_B="${PDE_LAMBDA_B:-1e-4}"
PDE_LAMBDA_W="${PDE_LAMBDA_W:-1e-4}"
PDE_LAMBDA_BINARY="${PDE_LAMBDA_BINARY:-0}"
PDE_SPARSE_GATE_INIT="${PDE_SPARSE_GATE_INIT:-0.5}"
PDE_GATE_THRESHOLD="${PDE_GATE_THRESHOLD:-0.4}"
PDE_ALPHA_THRESHOLD="${PDE_ALPHA_THRESHOLD:-1e-5}"
PDE_W_THRESHOLD="${PDE_W_THRESHOLD:-0.1}"
PDE_TERM_THRESHOLD="${PDE_TERM_THRESHOLD:-0.0}"
PDE_COEFFICIENT_THRESHOLD="${PDE_COEFFICIENT_THRESHOLD:-0.0}"
PDE_PRUNE_METRIC="${PDE_PRUNE_METRIC:-stridge}"
PDE_CONTRIBUTION_THRESHOLD="${PDE_CONTRIBUTION_THRESHOLD:-0.2}"
PDE_CANDIDATE_CONTRIBUTION_THRESHOLD="${PDE_CANDIDATE_CONTRIBUTION_THRESHOLD:-0.01}"
PDE_STLSQ_THRESHOLD="${PDE_STLSQ_THRESHOLD:-0.03}"
PDE_STRIDGE_LAMBDA="${PDE_STRIDGE_LAMBDA:-1e-5}"
PDE_DERIVATIVE_METHOD="${PDE_DERIVATIVE_METHOD:-finite_difference}"
PDE_FD_BOUNDARY="${PDE_FD_BOUNDARY:-5}"

# 真实 PDE 系数，用于计算项识别和系数误差。
# 默认会根据 DATA 自动选择；显式设置 TRUE_PDE_JSON 时优先使用手动值。
TRUE_PDE_JSON="${TRUE_PDE_JSON:-}"
SAVE_MODEL="${SAVE_MODEL:-0}"

if [[ -z "${TRUE_PDE_JSON}" ]]; then
  case "${DATA}" in
    "burgers_sine")
      # u_t + u*u_x - 0.1*u_xx = 0
      TRUE_PDE_JSON='{"u*u_x": -1.0, "u_xx": 0.1}'
      ;;
    "KdV-PINN")
      # u_t + u*u_x + 0.0025*u_xxx = 0
      TRUE_PDE_JSON='{"u*u_x": -1.0, "u_xxx": -0.0025}'
      ;;
    "Allen_Cahn")
      # u_t - 0.003*u_xx - u + u^3 = 0
      TRUE_PDE_JSON='{"u_xx": 0.003, "u": 1.0, "u^3": -1.0}'
      ;;
    "advection1d")
      # u_t + 0.1u_x = 0
      TRUE_PDE_JSON='{"u_x": -0.1}'
      ;;
    “KS”)
      # u_t + u*u_x + u_xx + u_xxxx = 0
      TRUE_PDE_JSON='{"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0}'
      ;;
    "convection-diffusion")
      # u_t + u_x - 0.25*u_xx = 0
      TRUE_PDE_JSON='{"u_x": -1.0, "u_xx": 0.25}'
      ;;
  esac
fi

cmd=(
  "${PYTHON_BIN}" -u main.py
  --data "${DATA}"
  --input_path "${INPUT_PATH}"
  --noise-std "${NOISE_STD}"
  --test-split "${TEST_SPLIT}"
  --num-hidden-layers "${NUM_HIDDEN_LAYERS}"
  --neurons-per-layer "${NEURONS_PER_LAYER}"
  --activation "${ACTIVATION}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --lambda-reg "${LAMBDA_REG}"
  --seed "${SEED}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${MODEL_PATH}" ]]; then
  cmd+=(--model-path "${MODEL_PATH}")
fi

if [[ "${FORCE_RETRAIN}" == "1" ]]; then
  cmd+=(--force-retrain)
fi

if [[ "${SAVE_MODEL}" == "1" ]]; then
  cmd+=(--save-model)
fi

if [[ "${DISCOVER_PDE}" == "1" ]]; then
  cmd+=(
    --discover-pde
    --pde-max-order "${PDE_MAX_ORDER}"
    --pde-rank "${PDE_RANK}"
    --pde-dense-epochs "${PDE_DENSE_EPOCHS}"
    --pde-sparse-epochs "${PDE_SPARSE_EPOCHS}"
    --pde-learning-rate "${PDE_LEARNING_RATE}"
    --pde-lambda-ridge "${PDE_LAMBDA_RIDGE}"
    --pde-lambda-g "${PDE_LAMBDA_G}"
    --pde-lambda-alpha "${PDE_LAMBDA_ALPHA}"
    --pde-lambda-b "${PDE_LAMBDA_B}"
    --pde-lambda-w "${PDE_LAMBDA_W}"
    --pde-lambda-binary "${PDE_LAMBDA_BINARY}"
    --pde-sparse-gate-init "${PDE_SPARSE_GATE_INIT}"
    --pde-gate-threshold "${PDE_GATE_THRESHOLD}"
    --pde-alpha-threshold "${PDE_ALPHA_THRESHOLD}"
    --pde-w-threshold "${PDE_W_THRESHOLD}"
    --pde-term-threshold "${PDE_TERM_THRESHOLD}"
    --pde-coefficient-threshold "${PDE_COEFFICIENT_THRESHOLD}"
    --pde-prune-metric "${PDE_PRUNE_METRIC}"
    --pde-contribution-threshold "${PDE_CONTRIBUTION_THRESHOLD}"
    --pde-stlsq-threshold "${PDE_STLSQ_THRESHOLD}"
    --pde-stridge-lambda "${PDE_STRIDGE_LAMBDA}"
    --pde-derivative-method "${PDE_DERIVATIVE_METHOD}"
    --pde-fd-boundary "${PDE_FD_BOUNDARY}"
  )

  if [[ -n "${PDE_RANKS}" ]]; then
    cmd+=(--pde-ranks "${PDE_RANKS}")
  fi

  if [[ -n "${PDE_CANDIDATE_CONTRIBUTION_THRESHOLD}" ]]; then
    cmd+=(--pde-candidate-contribution-threshold "${PDE_CANDIDATE_CONTRIBUTION_THRESHOLD}")
  fi

  if [[ -n "${TRUE_PDE_JSON}" ]]; then
    cmd+=(--true-pde-json "${TRUE_PDE_JSON}")
  fi
fi

cmd+=("$@")

printf 'Running command:\n'
printf '  %q' "${cmd[@]}"
printf '\n\n'

exec "${cmd[@]}"
