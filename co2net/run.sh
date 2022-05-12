#---------------------------------------------------------------------------------------------------
# THUMOS14 Training
CUDA_VISIBLE_DEVICES=0 python main.py --dataset-name Thumos14reduced --path-dataset path/to/Thumos14 --num-class 20 --max-seqlen 500 --use-model CO2 --max-iter 5000 --weight_decay 0.001 --model-name CO2_3552

# THUMOS14 Testing
CUDA_VISIBLE_DEVICES=0 python test.py --dataset-name Thumos14reduced --path-dataset path/to/Thumos14 --num-class 20 --max-seqlen 500 --use-model CO2 --model-name CO2_3552

#---------------------------------------------------------------------------------------------------
#ActivityNet Training
CUDA_VISIBLE_DEVICES=0 python main.py --dataset-name ActivityNet1.2 --path-dataset path/to/ActivityNet1.2 --num-class 100 --max-seqlen 60 --dataset AntSampleDataset --use-model ANT_CO2 --max-iter 22000 --model-name ANT_CO2_3552

# AcitivityNet1.2 Testing
CUDA_VISIBLE_DEVICES=0 python test.py --dataset-name ActivityNet1.2 --path-dataset path/to/ActivityNet1.2 --num-class 100 --max-seqlen 60 --dataset AntSampleDataset --use-model ANT_CO2 --model-name ANT_CO2_3552



