import sys
sys.path.append('.')
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config
import wandb

from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset


def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, use_wandb)
    else:
        coach = SingleIDCoach(dataloader, use_wandb)

    coach.train()

    return global_config.run_name



def export_updated_pickle(new_G,model_id):
  print("Exporting large updated pickle based off new generator and ffhq.pkl")
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    d = pickle.load(f)
    old_G = d['G_ema'].cuda() ## tensor
    old_D = d['D'].eval().requires_grad_(False).cpu()

  tmp = {}
  tmp['G_ema'] = old_G.eval().requires_grad_(False).cpu()# copy.deepcopy(new_G).eval().requires_grad_(False).cpu()
  tmp['G'] = new_G.eval().requires_grad_(False).cpu() # copy.deepcopy(new_G).eval().requires_grad_(False).cpu()
  tmp['D'] = old_D
  tmp['training_set_kwargs'] = None
  tmp['augment_pipe'] = None


  with open(f'{paths_config.checkpoints_dir}/stylegan2_custom_512_pytorch.pkl', 'wb') as f:
      pickle.dump(tmp, f)

if __name__ == '__main__':
    # run_PTI(run_name='', use_wandb=False, use_multi_id_training=False)
    paths_config.input_data_path = '/home/zhizizhang/Documents2/projects/PTI/test/'
    model_id = run_PTI(use_wandb=False, use_multi_id_training=False)
    print(model_id)
    image_name = 'test'
    import pickle
    import torch
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()  #// this grabs the pickle for ffhq file
        
    with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda() #// and htis is grabbing the updated model_MWVZTEZFDDJB_1.pt

    export_updated_pickle(new_G,model_id)