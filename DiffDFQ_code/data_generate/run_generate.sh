for g in 1 2 3 4
do
CUDA_VISIBLE_DEVICES=2 python3 generate_data.py 		\
		--model=resnet18 			\
		--batch_size=256 		\
		--test_batch_size=512 \
		--group=$g \
		--targetPro=0.9 \
		--cosineMargin=0.3 \
		--cosineMargin_upper=0.8 \
		--augMargin=0.5 \
		--save_path_head=./hardsample/resnet18/0.001/3bit \
		--qw=3\
		--qa=3
done