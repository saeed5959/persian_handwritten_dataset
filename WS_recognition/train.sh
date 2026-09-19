TRAIN_XML=${1:-dataset/img_xml}

ls "$TRAIN_XML"/*.xml | shuf --random-source=<(yes) > all.lst
N=$(wc -l < all.lst); NV=$(( N / 5 ))
head -n "$NV"        all.lst > val.lst
tail -n +$((NV + 1)) all.lst > train.lst
echo "train=$(wc -l < train.lst)  val=$(wc -l < val.lst)"

ketos train \
  -f page -t train.lst -e val.lst \
  -o checkpoints \
  --load /home/saeed/.local/share/htrmopo/230a3928-733e-5524-baa5-f89ba9b9eb70/all_arabic_scripts.mlmodel \
  --resize new \
  -B 8 -r 0.0001 --augment \
  -w 0 -q fixed --freq 1.0 -N 40 --gradient-clip-val 1 \
  --normalization NFD
