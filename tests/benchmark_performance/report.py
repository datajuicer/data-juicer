import wandb
import fire
import os
import json
import yaml
import regex as re
from loguru import logger

PROJECT = 'Data-Juicer Reports'
RUN_NAME = 'Performance Benchmark -- %s'
MODALITIES = {'text', 'image', 'video', 'audio'}
DIFF_TH = 0.1

def get_run_id(project, run_name, entity='dail'):
    api = wandb.Api()
    runs = api.runs(path=f'{entity}/{project}')
    for run in runs:
        if run.name == run_name:
            return run.id
    return ''

def init_run(modality, config=None):
    # get the run object for specified modality
    # if it's not existed, create one
    # if it's existed, get the run id and resume from it
    run_id = get_run_id(PROJECT, RUN_NAME % modality)
    if run_id == '':
        # no existing run, create one
        run = wandb.init(project=PROJECT,
                         config=config,
                         tags=['performance benchmark', modality],
                         name=RUN_NAME % modality)
        run_id = get_run_id(PROJECT, RUN_NAME % modality)
    else:
        run = wandb.init(project=PROJECT,
                         id=run_id,
                         resume='must')
    return run, run_id

def main():
    wandb.login()
    for modality in MODALITIES:
        logger.info(f'--------------- {modality} ---------------')
        work_dir = f'outputs/performance_benchmark_{modality}/'

        # read config
        with open(os.path.join(work_dir, f'{modality}.yaml')) as fin:
            config = yaml.load(fin, yaml.FullLoader)

        # init the wandb run
        run, run_id = init_run(modality, config)

        # collect results from logs
        log_pt = r'export_(.*?)_time_(\d*?).txt'
        log_dir = os.path.join(work_dir, 'log')
        log_files = os.listdir(log_dir)
        log_file = None
        for fn in log_files:
            if re.match(log_pt, fn):
                log_file = fn
                break
        if log_file is None:
            logger.warning('No log files found.')
            exit()
        log_file = os.path.join(log_dir, log_file)
        with open(log_file) as fin:
            log_content = fin.read()
        op_pt = r'OP \[(.*?)\] Done in (.*?)s'
        total_pt = r'All OPs are done in (.*?)s'
        op_data = re.findall(op_pt, log_content)
        ops = [it[0] for it in op_data]
        total_data = re.findall(total_pt, log_content)

        res = dict(op_data)
        res['total_time'] = total_data[0]
        res = {key: {'time': float(res[key])} for key in res}

        # collect resource utilization from monitor logs
        monitor_file = os.path.join(work_dir, 'monitor', 'monitor.json')
        with open(monitor_file) as fin:
            monitor_res = json.load(fin)
        assert len(monitor_res) == len(ops)
        for op, resource_util_dict in zip(ops, monitor_res):
            res[op].update(resource_util_dict['resource_analysis'])

        # upload results and finish the run
        upload_res = {
            modality: res
        }
        run.log(upload_res)
        run.finish()

        # compare with the last run
        api = wandb.Api()
        api_run = api.run(f'{PROJECT}/{run_id}')
        run_history = api_run.history()
        if len(run_history) < 2:
            continue
        last_record = run_history.iloc[-2]

        for op_name, time in op_data:
            last_time = last_record[f'{modality}.{op_name}.time']
            this_time = res[op_name]['time']
            dif = (this_time - last_time) / last_time
            if dif > 0.1:
                logger.warning(f'Time cost for OP {[op_name]} increased by '
                               f'{dif * 100}% (> 10%). Before-{last_time} vs. '
                               f'Now-{this_time}')
            else:
                logger.info(f'Time cost for OP {[op_name]} increased by '
                            f'{dif * 100}%. Before-{last_time} vs. '
                            f'Now-{this_time}')
        last_total = last_record[f'{modality}.total_time.time']
        this_total = res['total_time']['time']
        dif_total = (this_total - last_total) / last_total
        if dif_total > 0.1:
            logger.warning(f'Total time cost increased by {dif_total * 100}% '
                           f'(> 10%). Before-{last_total} vs. '
                           f'Now-{this_total}')
        else:
            logger.info(f'Total time cost increased by {dif_total * 100}%. '
                        f'Before-{last_total} vs. Now-{this_total}')


if __name__ == '__main__':
    fire.Fire(main)
