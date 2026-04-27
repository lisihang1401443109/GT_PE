import yaml
import copy

def create_job(template_path, new_path, job_name, dataset_name, completion_count):
    with open(template_path, 'r') as f:
        job = list(yaml.safe_load_all(f))[0]
    
    job['metadata']['name'] = job_name
    job['spec']['completions'] = completion_count
    job['spec']['parallelism'] = min(8, completion_count)
    
    container = job['spec']['template']['spec']['containers'][0]
    script = container['args'][0]
    
    # Adjust the script to focus on the specific dataset and correct index mapping
    new_script = script.replace(
        '# Index 0-7: ZINC, 8-15: VOC',
        f'# Partitioned Job for {dataset_name}'
    )
    
    if dataset_name.lower() == 'zinc':
        new_script = new_script.replace(
            'DATASET_IDX=$(( $JOB_COMPLETION_INDEX / 8 ))',
            'DATASET_IDX=0'
        )
    else:
        new_script = new_script.replace(
            'DATASET_IDX=$(( $JOB_COMPLETION_INDEX / 8 ))',
            'DATASET_IDX=1'
        )
    
    container['args'][0] = new_script
    
    with open(new_path, 'w') as f:
        yaml.dump(job, f, default_flow_style=False)

template = 'nautilus/grit_benchmarks.yaml'
create_job(template, 'nautilus/grit_benchmarks_zinc.yaml', 'grit-benchmarks-zinc', 'ZINC', 8)
create_job(template, 'nautilus/grit_benchmarks_voc.yaml', 'grit-benchmarks-voc', 'VOC', 8)
