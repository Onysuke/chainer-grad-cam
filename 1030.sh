$INPUT_IMG = '/var/docker/share/hyodo/20181017_gmm-eval-6/anomaly/20171021182405.jpg'
$OUT_NAME = 'gcam_20171021182405.jpg'
$GPU_ID = 1
$ARCH = 'my_model'
$LABEL = 1 #挙上
$WEIGHT_NAME="/mnt/aoni02/hyodo/1006_garbage_4class_no_normalize/1006/model_epoch-20_0.553070148688975-0.7695071876198631-0.7922506408677215-0.6443452341925531.npz"


python run.py --input $INPUT_IMG --output $OUT_NAME \
              --label $LABEL --gpu $GPU_ID \
              --arch $ARCH --model $WEIGHT_NAME
