python train.py \
  # -m torch.distributed.run \
  # --nproc_per_node 2 \
  # --master_port 1 train.py \ 
  # --device 0,1 \
  # --data data/licence.yaml \ "please refer to these files in order to modify configurations"
  # --cfg models/yolov5s.yaml \
  --epochs 60 \
  --batch 16 \
  --weights yolov5s.pt \
  --project my-awesome-project \
  --bbox_interval 1 \
  --save-period 1 \
  --hyp /home/percv-d0/dyn/yolov5/data/hyps/hyp.scratch.yaml