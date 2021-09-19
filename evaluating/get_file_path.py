# generating testing scripts
# we need this step because we didn't do testing together with training
# otherwise this step is not necessary 
from os import listdir
from os.path import join
import glob

# some of the pytorch_model.bin have a longer path, and some shorter
# we use the number in the dictionary to specify the path length
dates = {'msdialog':['07-08', 2]}#'msdialog':['07-28', 5], 'mantis':['08-13', 3],'msdialog_base':['05-29', 5], 'msdialog_inverse':['06-08', 5]
res = {}
prefix = "d:/graduation/transformers_cl_reproduce/"
for k, v in dates.items():
    full = prefix + v[0] + '/*'*v[1] + '/pytorch_model.bin'
    paths = glob.glob(full)

    model_paths = []
    output_paths = []
    for path in paths:
        p_list = path.split('\\')[1:-1]
        print(p_list)
        if (p_list[0].split("_"))[2] == "preds":
            model_path = "drive/MyDrive/transformers_cl/" + "/".join(p_list)
            model_paths.append(model_path)
            output_path = "drive/MyDrive/transformers_cl/" + p_list[0] + '_test'
            output_paths.append(output_path)
    res[k] = [model_paths, output_paths]


task_names = {"msdialog":"ms_v2", "mantis":"mantis_10"}
datasets = {"msdialog":"MSDialog", "mantis":"Mantis"}
for key, _ in dates.items():
   #print(key)
    task_name = task_names[key]
    dataset = datasets[key]
    model_paths, output_paths = res[key]
    #print(len(model_path))
    script_templates = []
    for i in range(len(model_paths)):
        model_path = model_paths[i]
        output_path = output_paths[i]
        script_template = f"""!python drive/MyDrive/transformers_cl/run_glue_test.py \\\n\t\t--model_type bert \\\n\t\t--config_name bert-base-uncased \\\n\t\t--tokenizer_name bert-base-uncased\\\n\t\t--model_name_or_path {model_path}\\\n\t\t--task_name  {task_name} \\\n\t\t--do_eval \\\n\t\t--do_lower_case \\\n\t\t--data_dir drive/MyDrive/{dataset} \\\n\t\t--max_seq_length 128 \\\n\t\t--per_gpu_eval_batch_size=64   \\\n\t\t--per_gpu_train_batch_size=64   \\\n\t\t--output_dir {output_path} \\"""
        with open(key + '.txt', 'a+') as fh:
            fh.write(f"{script_template}\n\n")
