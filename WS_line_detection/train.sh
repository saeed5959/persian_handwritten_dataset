#!/usr/bin/env bash
# Fine-tune kraken 7.x baseline segmenter.
#   dataset/img      dataset/xml       -> train + validation
#   dataset_test/img dataset_test/xml  -> held-out test, used once by segtest
set -euo pipefail

TRAIN_XML=${1:-dataset/img_xml}
TEST_XML=${2:-dataset_test/img_xml}
CKPT=${3:-checkpoints}
DEV=${4:-cuda:0}

# ---- 1. manifests: one XML path per line (format_type=page pulls the image
#         via Page/@imageFilename, resolved relative to the XML's directory) ----
ls "$TRAIN_XML"/*.xml | shuf --random-source=<(yes) > all.lst
N=$(wc -l < all.lst); NV=$(( N / 5 ))
head -n "$NV"        all.lst > val.lst
tail -n +$((NV + 1)) all.lst > train.lst
ls "$TEST_XML"/*.xml > test.lst
echo "train=$(wc -l < train.lst)  val=$(wc -l < val.lst)  test=$(wc -l < test.lst)"

# ---- 2. locate a pretrained base segmenter (optional) ----
BLLA=$(python3 - <<'PY'
import glob, os
try:
    import kraken
    d = os.path.dirname(kraken.__file__)
    c = glob.glob(d + "/blla.*safetensors") + glob.glob(d + "/blla.*mlmodel") + glob.glob(d + "/blla.*")
    print(c[0] if c else "")
except Exception:
    print("")
PY
)
if [ -n "$BLLA" ]; then
  echo "base model: $BLLA"
  LOAD_LINE="  load: $BLLA"
  RESIZE_LINE="  resize: new"
else
  echo "no blla found -> training from scratch (try: kraken get blla)"
  LOAD_LINE=""
  RESIZE_LINE=""
fi

# ---- 3. experiment file ----
cat > seg_config.yml <<YML
# global options live at the top level
device: $DEV
precision: bf16-mixed
num_workers: 0
num_threads: 1
seed: 42

segtrain:
  training_data:
    - train.lst
  evaluation_data:
    - val.lst
  format_type: page
  checkpoint_path: $CKPT
  weights_format: safetensors

$LOAD_LINE
$RESIZE_LINE

  # Annotation sits ~35% down the line box -> closest to a central line.
  # false = baseline (bottom of letter bodies), true = topline, null = centerline
  topline: null

  augment: true

  # merge every line type into one class; same for regions.
  # set region_class_mapping to [] to train baselines only.

  quit: fixed
  epochs: 25
  min_epochs: 5
  lag: 15
  lrate: 1e-4
  weight_decay: 1e-5
  schedule: cosine
  cos_t_max: 100
  cos_min_lr: 1e-5
  gradient_clip_val: 1.0
  warmup: 50   
YML

# ---- 4. train ----
ketos --config seg_config.yml segtrain

# ---- 5. held-out test, once ----
#BEST=$(ls -t "$CKPT"/*best*.safetensors "$CKPT"/*.safetensors 2>/dev/null | head -1)
#echo "best weights: $BEST"
#ketos --device "$DEV" segtest -m "$BEST" -e test.lst -f page  --test-class-mapping-mode canonical --bl-tol 10

# ---- 6. inference over the 10k corpus ----
# kraken -I 'corpus/*.jpg' -f image -o '.xml' segment -bl -i "$BEST"
