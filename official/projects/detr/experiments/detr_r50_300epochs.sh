#!/bin/bash
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=99999999999999999999

# train & eval, runnable!
# task.train_data.global_batch_size=64,task.validation_data.global_batch_size=64,trainer.checkpoint_interval=10,trainer.validation_interval=10
export PYTHONPATH="${PYTHONPATH}:/home/neoxygong/project/model_garden"

export TPU_NAME=local
export PARAMS=task.init_checkpoint="official/projects/detr/weights/resnet50_imagenet/ckpt-28080"
export PARAMS=$PARAMS,trainer.train_steps=554400
export PARAMS=$PARAMS,trainer.optimizer_config.learning_rate.stepwise.boundaries="[369600]"
export PARAMS=$PARAMS,trainer.preemption_on_demand_checkpoint=False
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu

python3 official/projects/detr/train.py \
  --experiment=detr_coco \
  --mode=train_and_eval \
  --model_dir=official/projects/detr/output/detr_r50_300epochs \
  --tpu=$TPU_NAME \
  --params_override=$PARAMS


# train & eval, bs 48
# task.train_data.global_batch_size=64,task.validation_data.global_batch_size=64,trainer.checkpoint_interval=10,trainer.validation_interval=10
export PYTHONPATH="${PYTHONPATH}:/home/neoxygong/project/model_garden"

export TPU_NAME=local
export PARAMS=task.init_checkpoint="official/projects/detr/weights/resnet50_imagenet/ckpt-28080"
export PARAMS=$PARAMS,trainer.train_steps=554400
export PARAMS=$PARAMS,task.train_data.global_batch_size=48
export PARAMS=$PARAMS,task.validation_data.global_batch_size=64
export PARAMS=$PARAMS,trainer.optimizer_config.learning_rate.stepwise.boundaries="[369600]"
export PARAMS=$PARAMS,trainer.optimizer_config.learning_rate.stepwise.values='[0.000075,0.0000075]'
export PARAMS=$PARAMS,trainer.preemption_on_demand_checkpoint=False
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu

python3 official/projects/detr/train.py \
  --experiment=detr_coco \
  --mode=train_and_eval \
  --model_dir=official/projects/detr/output/detr_r50_300epochs_bs48 \
  --tpu=$TPU_NAME \
  --params_override=$PARAMS

# eval with model_dir, runnable, 0 AP
export PYTHONPATH="${PYTHONPATH}:/home/neoxygong/project/model_garden"

export TPU_NAME=local
export PARAMS=task.validation_data.global_batch_size=64
export PARAMS=$PARAMS,runtime.distribution_strategy=tpu
python3 official/projects/detr/train.py \
  --experiment=detr_coco \
  --mode=eval \
  --model_dir=official/projects/detr/weights/detr_resnet_50_500/ckpt-924000 \
  --tpu=$TPU_NAME \
  --params_override=$PARAMS

