export PYTHONPATH="${PYTHONPATH}:/home/neoxygong/project/model_garden"

NUN_QUERIES=300
export TPU_NAME=local
export PARAMS=task.init_checkpoint="official/projects/detr/weights/resnet50_imagenet/ckpt-28080"
export PARAMS=$PARAMS,trainer.preemption_on_demand_checkpoint=False
export PARAMS=$PARAMS,trainer.validation_interval=5000
export PARAMS=$PARAMS,task.validation_data.global_batch_size=32
export PARAMS=$PARAMS,task.model.num_queries=$NUN_QUERIES
export PARAMS=$PARAMS,task.train_data.max_num_boxes=$NUN_QUERIES
export PARAMS=$PARAMS,task.validation_data.max_num_boxes=$NUN_QUERIES
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu

python3 official/projects/detr/train.py \
  --experiment=dino_coco \
  --mode=train_and_eval \
  --model_dir=official/projects/detr/output/dino_r50_50epochs_bs32 \
  --tpu=$TPU_NAME \
  --params_override=$PARAMS

# eval with model_dir, runnable, 0 AP
export PYTHONPATH="${PYTHONPATH}:/home/neoxygong/project/model_garden"

export TPU_NAME=local
export PARAMS=task.validation_data.global_batch_size=64
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu
python3 official/projects/detr/train.py \
  --experiment=dino_coco \
  --mode=eval \
  --model_dir=official/projects/detr/output/dino_r50_50epochs_bs32 \
  --tpu=$TPU_NAME \
  --params_override=$PARAMS
