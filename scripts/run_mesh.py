import os

scenes = ['']
factors = ['1']
data_devices = ['cuda']
data_base_path=''
out_base_path=''
out_name='test'
gpu_id=

for id, scene in enumerate(scenes):

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} --skip_test'
    print(cmd)
    os.system(cmd)