"""

The pre-requisite is to run the following command before invoking this binary:
The following command downloads the task config to the task folder 
>>> !PYTHONPATH=/content/jiant python jiant/jiant/scripts/download_data/runscript.py \
    download \
    --tasks mrpc \
    --output_path=/home/ugrads/nonmajors/mehrdadk/tmp/content/tasks/
"""

import os
import sys

import jiant.jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.jiant.proj.main.export_model as export_model
import jiant.jiant.proj.main.scripts.configurator as configurator
import jiant.jiant.proj.main.runscript as main_runscript
import jiant.jiant.shared.caching as caching
import jiant.jiant.utils.python.io as py_io
import jiant.jiant.utils.display as display

import jiant.jiant.scripts.download_data.runscript as runscript

_TASK_NAME = "mrpc"
_MODEL_NAME = 'albert-base-v2'

def downloa_tasks(task_name=_TASK_NAME):
    runscript.download_data(task_names, output_base_path)


def run_and_eval(model_name=_MODEL_NAME, task_name=_TASK_NAME):
    root_dir = "/home/ugrads/nonmajors/mehrdadk/tmp"
    task_dir = f"{root_path}/content/tasks/configs"
    model_dir = f"{root_dir}/models/{model_name}/model"

    export_model.export_model(
        hf_pretrained_model_name_or_path=model_name,
        output_base_path=model_dir,
    )

    # Tokenize and cache each task
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"{task_dir}/{task_name}_config.json",
        hf_pretrained_model_name_or_path="albert-base-v2",
        output_dir=f"{root_path}/tmp/cache/{task_name}",
        phases=["train", "val"],
    ))

    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=task_dir,
        task_cache_base_path=f"{root_dir}/tmp/cache",
        train_task_name_list=["mrpc"],
        val_task_name_list=["mrpc"],
        train_batch_size=1,
        eval_batch_size=1,
        epochs=3,
        num_gpus=1,
    ).create_config()

    os.makedirs(f"{root_path}/tmp/run_configs/", exist_ok=True)
    py_io.write_json(jiant_run_config, f"{root_path}/run_configs/mrpc_run_config.json")
    display.show_json(jiant_run_config)

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path= f"{root_path}/run_configs/mrpc_run_config.json",
        output_dir= f"{root_dir}/runs/mrpc",
        hf_pretrained_model_name_or_path=model_name,
        model_path= f"{model_dir}/model/model.p",
        model_config_path= f"{model_dir}/model/config.json",
        learning_rate=1e-5,
        eval_every_steps=500,
        do_train=True,
        do_val=True,
        do_save=True,
        force_overwrite=True,
    )

    runner = setup_runner_from_config(args=run_args)

    main_runscript.run_loop(run_args, runner=runner)
