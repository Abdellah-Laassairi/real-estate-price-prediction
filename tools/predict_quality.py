import os

test_dir = 'data/reduced_images/test'
train_dir = 'data/reduced_images/train'

test_images = [x[0] for x in os.walk(test_dir)]
train_images = [x[0] for x in os.walk(train_dir)]

for i in test_images[1:]:
    cmd = f'image-quality-assessment/predict  \
    --docker-image nima-cpu \
    --base-model-name MobileNet \
    --weights-file /home/odeck/Desktop/data-challenge-2022/image-quality-assessment/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 \
    --image-source /home/odeck/Desktop/data-challenge-2022/{i}'

    os.system(cmd)
    print(i)
    print(cmd)
