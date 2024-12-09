# Finetune Qwen2-VL

This repo is derived from [finetune-Qwen2-VL](https://github.com/zhangfaen/finetune-Qwen2-VL) and [SeeClick](https://github.com/njucckevin/SeeClick).


- [x] Support Qwen2-VL-7B-Instruct
- [x] Support LoRA Training and Customization
- [x] Support Lazy Dataset Loading

## Requirements

```bash
pip install -r requirements.txt
```

## with LoRA Training

```bash
bash finetune_qwen2_vl.sh --save-name SFT_Qwen2_VL_1 --max-length 512 --micro-batch-size 4 --save-interval 500 
    --train-epochs 3 --nproc-per-node 2 --data-path train_data/data.json --learning-rate 3e-5 
    --gradient-accumulation-steps 8 --qwen-ckpt xxxx/Qwen2-VL-7B-Instruct --pretrain-ckpt xxxx/Qwen2-VL-7B-Instruct
    --save-path xxxx/ckpt --use-lora
```

## w.o. LoRA Training

```bash
bash finetune_qwen2_vl.sh --save-name SFT_Qwen2_VL_1 --max-length 512 --micro-batch-size 4 --save-interval 500 
    --train-epochs 3 --nproc-per-node 2 --data-path train_data/data.json --learning-rate 3e-5 
    --gradient-accumulation-steps 8 --qwen-ckpt xxxx/Qwen2-VL-7B-Instruct --pretrain-ckpt xxxx/Qwen2-VL-7B-Instruct
    --save-path xxxx/ckpt
```

You can also retrain the model by adjust the parameter `--pretrain-ckpt`.



## License
This project incorporates specific datasets and checkpoints governed by their original licenses. Users are required to adhere to all terms of these licenses. No additional restrictions are imposed by this project beyond those specified in the original licenses.

