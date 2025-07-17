import os

scenes = ['']
factors = ['1']
data_devices = ['cuda']
data_base_path=''
out_base_path=''
out_name='test'
gpu_id=

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} -r{factors[id]} --data_device {data_devices[id]}'
    print(cmd)
    os.system(cmd)