

```bash
ln -s /data2/sun/data ./data
vim mixup.yml
CUDA_VISIBLE_DEVICES=0 nohup python train.py --opt ./mixup.yml >log.txt 2>&1 &
```

