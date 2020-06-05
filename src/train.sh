cd ../
python -m src.run --train_test train\
	          --data_root data \
		  --logging_root log \
		  --checkpoint_dir checkpoints \
		  --experiment_name 0_01M \
		  --max_epoch 50 \
		  --batch_size 32 \
		  --test_output_dir test_out \
		  --sf 1000 \
		  --vf 1000
