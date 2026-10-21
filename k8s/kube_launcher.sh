#!/bin/bash

# Launch one Kubernetes Job per model id over a range of ids from a hyperparameter file.
#
#   ./k8s/kube_launcher.sh ./hparams/hpo_1.json 0 29
#
# The template must contain these placeholders, which are substituted per job:
#
#   {{JOB_NAME}}   metadata.name          e.g. hpo-1-007
#   {{MODEL_ID}}   env MODEL_ID value     e.g. 7      (matches 'id' in the params file)
#   {{RUN_NAME}}   env RUN_NAME value     e.g. hpo_1  (train.sh reads ./hparams/$RUN_NAME.json)
#
# Env:
#   TEMPLATE=<path>   job template to render (default: k8s/job_template.yaml)
#   DRY_RUN=1         render and print, submit nothing
#   ASSUME_YES=1      skip the confirmation prompt

set -euo pipefail

TEMPLATE="${TEMPLATE:-k8s/train_template.yml}"
DRY_RUN="${DRY_RUN:-0}"
ASSUME_YES="${ASSUME_YES:-0}"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  echo "usage: $0 <params-file> <start-id> <end-id>" >&2
  echo "   e.g: $0 ./hparams/hpo_1.json 0 29" >&2
  exit 2
}

[ "$#" -eq 3 ] || usage
PARAMS_FILE="$1"
START_ID="$2"
END_ID="$3"

# VALIDATE ARGS
[ -f "$PARAMS_FILE" ] || die "params file not found: $PARAMS_FILE"
[ -f "$TEMPLATE" ]    || die "template not found: $TEMPLATE (override with TEMPLATE=<path>)"
[[ "$START_ID" =~ ^[0-9]+$ ]] || die "start-id must be a non-negative integer, got '$START_ID'"
[[ "$END_ID"   =~ ^[0-9]+$ ]] || die "end-id must be a non-negative integer, got '$END_ID'"
[ "$START_ID" -le "$END_ID" ] || die "start-id ($START_ID) is greater than end-id ($END_ID)"

# RUN_NAME IS THE PARAMS BASENAME -- train.sh REBUILDS THE PATH AS ./hparams/$RUN_NAME.json,
# SO A FILE OUTSIDE hparams/ OR WITH A DIFFERENT NAME WOULD NOT RESOLVE INSIDE THE CONTAINER
RUN_NAME="$(basename "$PARAMS_FILE" .json)"
[ "$(basename "$(dirname "$PARAMS_FILE")")" = "hparams" ] \
  || die "params file must live in hparams/ so the container can find ./hparams/${RUN_NAME}.json"

# TEMPLATE MUST CARRY EVERY PLACEHOLDER -- OTHERWISE WE WOULD SUBMIT N IDENTICAL JOBS
for ph in JOB_NAME MODEL_ID RUN_NAME; do
  grep -q "{{${ph}}}" "$TEMPLATE" || die "template $TEMPLATE has no {{${ph}}} placeholder"
done

# IDS PRESENT IN THE PARAMS FILE (JSON LINES, ONE OBJECT PER MODEL)
declare -A HAS_ID=()
while read -r id; do HAS_ID["$id"]=1; done < <(
  python3 -c '
import json,sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        print(json.loads(line)["id"])
' "$PARAMS_FILE"
)
[ "${#HAS_ID[@]}" -gt 0 ] || die "no ids found in $PARAMS_FILE"

# SPLIT THE REQUESTED RANGE INTO WHAT EXISTS AND WHAT DOES NOT
TO_LAUNCH=()
MISSING=()
for ((id=START_ID; id<=END_ID; id++)); do
  if [ -n "${HAS_ID[$id]:-}" ]; then TO_LAUNCH+=("$id"); else MISSING+=("$id"); fi
done
[ "${#TO_LAUNCH[@]}" -gt 0 ] || die "none of the ids ${START_ID}..${END_ID} exist in $PARAMS_FILE"

# k8s OBJECT NAMES ARE RFC 1123 DNS LABELS: LOWERCASE ALPHANUMERIC AND '-' ONLY.
# RUN_NAME VALUES LIKE 'hpo_1' CONTAIN AN UNDERSCORE, WHICH THE API SERVER REJECTS.
sanitize() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-' \
    | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$//'
}
NAME_PREFIX="$(sanitize "$RUN_NAME")"
[ -n "$NAME_PREFIX" ] || die "run name '$RUN_NAME' sanitizes to an empty job name prefix"

# PLAN
echo "params file : $PARAMS_FILE"
echo "run name    : $RUN_NAME   (job names: ${NAME_PREFIX}-XXX)"
echo "template    : $TEMPLATE"
echo "ids         : ${#TO_LAUNCH[@]} job(s) over ${START_ID}..${END_ID}"
[ "${#MISSING[@]}" -gt 0 ] && echo "skipping    : ${#MISSING[@]} id(s) absent from the params file: ${MISSING[*]}"
if [ "$DRY_RUN" = "1" ]; then
  echo "context     : (dry run, nothing will be submitted)"
else
  echo "context     : $(kubectl config current-context 2>/dev/null || echo 'unknown')"
fi

# CONFIRM BEFORE PUSHING JOBS TO A CLUSTER
if [ "$DRY_RUN" != "1" ] && [ "$ASSUME_YES" != "1" ]; then
  read -r -p "Submit ${#TO_LAUNCH[@]} job(s) to the context above? [y/N] " reply
  case "$reply" in [yY]|[yY][eE][sS]) ;; *) echo "aborted"; exit 1;; esac
fi

# RENDER OUTSIDE THE REPO SO A STRAY MANIFEST CAN NEVER BE COMMITTED
RENDERED="$(mktemp -t "kube_launcher_${NAME_PREFIX}_XXXXXX.yaml")"
cleanup() { rm -f "$RENDERED"; }
trap cleanup EXIT

SUBMITTED=0
FAILED=()
for MODEL_ID in "${TO_LAUNCH[@]}"; do
  MODEL_ID_PADDED=$(printf "%03d" "$MODEL_ID")
  JOB_NAME="${NAME_PREFIX}-${MODEL_ID_PADDED}"

  # DNS LABELS CAP AT 63 CHARACTERS
  [ "${#JOB_NAME}" -le 63 ] || die "job name '$JOB_NAME' exceeds the 63 character limit"

  sed -e "s|{{JOB_NAME}}|${JOB_NAME}|g" \
      -e "s|{{MODEL_ID}}|${MODEL_ID}|g" \
      -e "s|{{RUN_NAME}}|${RUN_NAME}|g" \
      "$TEMPLATE" > "$RENDERED"

  # ANY PLACEHOLDER LEFT BEHIND MEANS THE TEMPLATE WANTS SOMETHING WE DO NOT SUBSTITUTE
  if grep -oE '\{\{[A-Za-z0-9_]+\}\}' "$RENDERED" | sort -u | grep . >/dev/null; then
    die "unsubstituted placeholder(s) in $TEMPLATE: $(grep -oE '\{\{[A-Za-z0-9_]+\}\}' "$RENDERED" | sort -u | tr '\n' ' ')"
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "--- $JOB_NAME (MODEL_ID=$MODEL_ID, RUN_NAME=$RUN_NAME) ---"
    cat "$RENDERED"
    SUBMITTED=$((SUBMITTED+1))
    continue
  fi

  # create, NOT apply: a Job spec is largely immutable, so a name collision should
  # fail loudly here rather than be patched into a confusing error by the API server
  if kubectl create -f "$RENDERED"; then
    SUBMITTED=$((SUBMITTED+1))
  else
    echo "error: failed to submit $JOB_NAME (id $MODEL_ID)" >&2
    cp "$RENDERED" "${RENDERED%.yaml}_${MODEL_ID_PADDED}_failed.yaml"
    echo "       rendered manifest kept at ${RENDERED%.yaml}_${MODEL_ID_PADDED}_failed.yaml" >&2
    FAILED+=("$MODEL_ID")
  fi
done

echo
if [ "$DRY_RUN" = "1" ]; then
  echo "dry run: rendered $SUBMITTED manifest(s), submitted none"
else
  echo "submitted $SUBMITTED of ${#TO_LAUNCH[@]} job(s)"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "failed ids: ${FAILED[*]}" >&2
  exit 1
fi
