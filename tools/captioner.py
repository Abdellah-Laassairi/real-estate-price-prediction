import glob
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from time import sleep

import pandas as pd
import requests
import torch
from PIL import Image
from rich import progress
from rich.console import Console
from tqdm import trange
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

console = Console()


def extract_features(data,
                     progress,
                     task_id,
                     processor_name='microsoft/git-base-coco',
                     model_name='microsoft/git-base-coco',
                     device='cuda'):

    device = f'{device}:{int(task_id)}'
    console.log(f'Loading preprocessor {processor_name} for task : {task_id}')
    processor = AutoProcessor.from_pretrained(processor_name)
    console.log(f'Loading model {model_name} for task : {task_id}')
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    console.log(f'Running inference for : {task_id}')

    results = []
    for i in range(len(data)):
        row = data.iloc[i]
        base_path = '../data/reduced_images/test/ann_'
        house_id_path = base_path + str(int(row['id_annonce']))

        house_images = glob.glob(f'{house_id_path}/*jpg')
        images = []
        for path_image in house_images:
            image = Image.open(path_image)
            images.append(image)

        # unconditional image captioning
        pixel_values = processor(images=images,
                                 return_tensors='pt').pixel_values.to(device)

        generated_ids = model.generate(pixel_values=pixel_values,
                                       max_length=50)
        generated_caption = processor.batch_decode(generated_ids,
                                                   skip_special_tokens=True)

        result = {}
        result['id_annonce'] = int(row['id_annonce'])
        result['features'] = generated_caption
        results.append(result)
        i += 1

        progress[task_id + 1] = {'progress': i + 1, 'total': len(data)}

    with open(f'../data/image_captions/result_test_{task_id}.json', 'w') as fp:
        json.dump(results, fp)

    console.log(f'Finished task {task_id}')


if __name__ == '__main__':

    y_train_raw = pd.read_csv('../data/tabular/y_train_OXxrJt1.csv')
    x_test_raw = pd.read_csv('../data/tabular/X_test_BEhvxAN.csv')
    sub = len(x_test_raw) // 3
    subsets = [x_test_raw[:sub], x_test_raw[sub:2 * sub], x_test_raw[2 * sub:]]

    for i, sub in enumerate(subsets):
        print(f'Task {i} : {len(sub)}')
    device = 'cuda'
    n_workers = 3

    with progress.Progress(
            '[progress.description]{task.description}',
            progress.BarColumn(),
            '[progress.percentage]{task.percentage:>3.0f}%',
            progress.TimeRemainingColumn(),
            progress.TransferSpeedColumn(),
            progress.TimeElapsedColumn(),
            refresh_per_second=1,  # bit slower updates
    ) as progress:
        #extract_features(subsets[0], progress, task_id=1, )
        futures = []  # keep track of the jobs
        with multiprocessing.Manager() as manager:
            # this is the key - we share some state between our
            # main process and our worker functions
            _progress = manager.dict()
            overall_progress_task = progress.add_task(
                '[green]All jobs progress:')

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for n in range(
                        n_workers):  # iterate over the jobs we need to run

                    task_id = progress.add_task(f'Task {n}', visible=False)
                    futures.append(
                        executor.submit(extract_features, subsets[n],
                                        _progress, task_id - 1))

                # monitor the progress:
                while (n_finished := sum([future.done() for future in futures
                                          ])) < len(futures):
                    progress.update(overall_progress_task,
                                    completed=n_finished,
                                    total=len(futures))
                    for task_id, update_data in _progress.items():
                        latest = update_data['progress']
                        total = update_data['total']
                        # update the progress bar for this task:
                        progress.update(
                            task_id,
                            completed=latest,
                            total=total,
                            visible=latest < total,
                        )

                # raise any errors:
                for future in futures:
                    future.result()
