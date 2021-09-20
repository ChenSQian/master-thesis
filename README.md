# Master Thesis
This is the repository of Chen Qian's master thesis: An Empirical Research on Dynamic CurriculumLearning in Information Retrieval.

## Abstract
Most humans need to spend nearly twenty years on well-organized training before fully functioning in human society. For many people, well-organized training means following a school curriculum, which prepares the knowledge in a meaningful order, usually from easier concepts to more complexconcepts. Providing the previously learned knowledge can accelerate the learning speed of newones. Resembling the human learning process, curriculum learning (CL) has been successfully applied in many machine learning fields. Given that lack of training data has become the bottleneckof many research questions. Curriculum learning is a promising research direction to tackle thisproblem, as it can improve model performance without extra computational cost and requirementsfor additional data. Also, there remains a research gap of CL in the field of Information retrieval (IR). We contribute to this research gap by doing empirical explorations with the state-of-the-art language model BERT in one of the complex IR tasks, conversation response ranking (CRR). The existing CL frameworks contain two main steps, first the difficulty criterion to arrange the data from easy to difficult, and second the speed criterion to control the pacing of transferring from easy to difficult data. However, the difficulty criterion or the speed criterion usually remains unchanged and static during the entire training process. For a more comprehensive exploration, we propose the dynamic rescoring method for the first step, and the noise method for the second step. The experimental results show: 1) the dynamic rescoring method has no exciting improvement overthe baseline CL methods, whereas the noise method can slightly outperform the baselines with a nonexcessive noise ratio; 2) the comparison of different settings informs us that a good difficulty criterion and a proper speed criterion are essential for CL to be effective.

## Curriculum Learning Framework
The curriculum learning frameworks, adopted from previous works, contain two main steps, first the difficulty criterion to arrange the datafrom easy to difficult, and second the speed criterion to control the pacing of transferring from easyto difficult data. The two steps are illustrated in the following figure.

<p align="center">
  <img width="500" alt="scoring_pacing" src="https://user-images.githubusercontent.com/56640848/133944675-56da538b-2ff9-4043-a446-8e097d85bf17.png">
</p>

## Dynamic Rescoring Method
For the difficulty criterion step, we design the dynamic rescoring methodto explore if  re-sorting  the  data  several  times  according  to  the  current  model  can  help  improve  the performance. The motivation of the dynamic rescoring method is that while the model is learning, its evaluation of the difficulty is also changing. A visualization of this method is in the following figure.

<p align="center">
  <img width="400" alt="dynamic_7" src="https://user-images.githubusercontent.com/56640848/133944635-6224b18d-d374-4f8b-b31c-fec576e6222a.png">
</p>  

## Noise Method
For the speed criterion, in addition to determining the speed by a pacing function, we combine the idea from previous work and apply the noise method. In brief, their vision is to allow the model to have more explorations in the early learning stages and gradually back to its learning goal. Likewise, in our work, by adding noise to the CL-scheduled data with a shrinking amount automatically adapted with the number of iterations, we explore if more flexibility can be beneficial. A visualization of this method is in the following figure.

<p align="center">
  <img width="500" alt="noise_overview" src="https://user-images.githubusercontent.com/56640848/133944627-e5e2a40a-8e7e-4513-a9b5-c7a44e5881de.png">
</p>

# Reproduce
## Set up Environment
We use BERT as the neural ranking model, and choose MSDialog and MANtIS as the conversation response ranking (CRR) task datasets.

install the packages in the requirements.txt

`!pip install -r ../requirements.txt`

## An Exmaple for Generating Scoring File

`!python drive/MyDrive/transformers_cl/run_glue_dynamic.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name  mantis_10 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --data_dir drive/MyDrive/transformer_rankers \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir drive/MyDrive/transformers_cl/mantis_10_bert_bm25_8/ \
    --save_aps \
    --eval_all_checkpoints \
    --logging_steps 1000 \
    `

# An Example for Training and Testing

`!python drive/MyDrive/transformers_cl/run_glue_noise.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name  ms_v2 \
    --do_train \
    --evaluate_during_training \
    --do_lower_case \
    --data_dir drive/MyDrive/MSDialog \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=64   \
    --per_gpu_train_batch_size=64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --seed 1\
    --output_dir drive/MyDrive/transformers_cl/ms_v2_root2_l_99_r_05_seed_1 \
    --logging_steps 100 \
    --curriculum_file  drive/MyDrive/MSDialog/preds_dif_run_seed_42 \
    --pacing_function root_2\
    --use_additive_cl \
    --eval_all_checkpoints \
    --invert_cl_values\
    --noise_lambda 0.99\
    --noise_difficult_ratio 0.5\
    `
