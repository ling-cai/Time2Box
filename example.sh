# CUDA_VISIBLE_DEVICES="1" python -u codes/run.py --cuda --do_test \
#   --data_path data/wikidata_toy_new/ --model BoxTransE -n 128 -b 1500 -d 400 -g 24 -a 1.0 \
#   -lr 0.0003 --max_steps 600000 --cpu_num 10 --test_batch_size 1 --center_reg 0.02 \
#   --geo box --task 1c.2i.3i.3i-2i --stepsforpath 1000 --offset_deepsets inductive --center_deepsets eleattention \
#   --print_on_screen --valid_steps 5000 --label "test-result_d400-128_notr"  --log_steps 1000 --activation relu \
#   --warm_up_steps 600000  --use_one_sample  --model_save_step 1000 --negative_sample_types tail-batch --predict_o \
#   --enumerate_time --init '/home/ling/Dynamic-Graph/query2box/logs/box/'

# CUDA_VISIBLE_DEVICES="0" python -u codes/run.py --cuda --do_train --do_valid --do_test \
#   --data_path data/wikidata_toy_new/ --model BoxTransE -n 128 -b 1500 -d 400 -g 24 -a 1.0 \
#   -lr 0.0003 --max_steps 600000 --cpu_num 10 --test_batch_size 1 --center_reg 0.02 \
#   --geo box --task 1c.2i.3i.3i-2i --stepsforpath 1000 --offset_deepsets inductive --center_deepsets eleattention \
#   --print_on_screen --valid_steps 10000 --label "test-result_d400-128-neg-2"  --log_steps 1000 --activation relu \
#   --warm_up_steps 600000  --use_one_sample  --model_save_step 1000 --negative_sample_types tail-batch.time_batch --predict_o --num_time_negatives 2 --time_score_weight 0.05\
#   --enumerate_time 


# ## Train ON WIKIDATA114k
CUDA_VISIBLE_DEVICES="1" python -u codes/run.py --cuda --do_test \
  --data_path data/WIKIDATA114k/ --model BoxTransE -n 128 -b 1500 -d 400 -g 24 -a 1.0 \
  -lr 0.0003 --max_steps 600000 --cpu_num 10 --test_batch_size 1 --center_reg 0.02 \
  --geo box --task 1c.2i.3i.3i-2i --stepsforpath 1000 --offset_deepsets inductive --center_deepsets eleattention \
  --print_on_screen --valid_steps 10000 --label "test"  --log_steps 1000 --activation relu \
  --warm_up_steps 600000  --use_one_sample  --model_save_step 1000 --negative_sample_types tail-batch.time_batch \
  --predict_o --num_time_negatives 8 --time_score_weight 0.05\
  --enumerate_time

# ## Train ON WIKIDATA12k
# CUDA_VISIBLE_DEVICES="0" python -u codes/run.py --cuda --do_train --do_valid --do_test \
#   --data_path data/WIKIDATA12k/ --model BoxTransE -n 128 -b 3000 -d 400 -g 9 -a 1.0 \
#   -lr 0.001 --max_steps 10000 --cpu_num 10 --test_batch_size 1 --center_reg 0.02 \
#   --geo box --task 2i.3i.3i-2i --stepsforpath 1000 --offset_deepsets inductive --center_deepsets eleattention \
#   --print_on_screen --valid_steps 500 --label "test"  --log_steps 1000 --activation relu \
#   --warm_up_steps 60 --use_one_sample  --model_save_step 1000 --negative_sample_types tail-batch --predict_o --time_smooth_weight 0.001 \
#   --enumerate_time