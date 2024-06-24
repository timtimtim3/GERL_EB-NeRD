# -*- encoding:utf-8 -*-
"""
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
import json
import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import Subset
import hydra
from omegaconf import DictConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Set GERL environment variable if not set
if "GERL" not in os.environ:
    os.environ["GERL"] = "/home/scur1584"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.dataset import TrainingDataset
from datasets.dataset import ValidationDataset
from models.gerl import Model, ModelDocEmb
# from utils.log_util import convert_omegaconf_to_dict
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch
from utils.eval_util import group_labels
from utils.eval_util import cal_metric


def find_free_port(start_port=9237, end_port=10000):
    import socket
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError("No free ports available")


def run(cfg: DictConfig, rank: int, device: torch.device, corpus_path: str):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    set_seed(cfg.training.seed)

    print("Worker %d is setting dataset ... " % rank)
    # Build Dataloader
    train_dataset = TrainingDataset(cfg.dataset, corpus_path)
    train_data_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)


    img_embeddings = None
    if cfg.training.use_img_embeddings:
        img_embeddings_path = cfg.dataset.img_embedding
        print(f"Loading image embeddings from: {img_embeddings_path}")
        img_embeddings = np.load(img_embeddings_path)
        img_embeddings = torch.tensor(img_embeddings, dtype=torch.float).to(device)
    
    doc_embeddings = None
    if cfg.training.use_doc_embeddings:
        doc_embeddings_path = cfg.dataset.doc_embedding
        print(f"Loading document embeddings from: {doc_embeddings_path}")
        doc_embeddings = np.load(doc_embeddings_path)
        doc_embeddings = torch.tensor(doc_embeddings, dtype=torch.float).to(device)
        model = ModelDocEmb(cfg, doc_embeddings=doc_embeddings, img_embeddings=img_embeddings)
    else:
        model = Model(cfg, img_embeddings=img_embeddings)

    # Build optimizer.
    steps_one_epoch = len(train_data_loader) // cfg.training.accumulate
    train_steps = cfg.training.epochs * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=1e-5)

    if isinstance(model, Model):
        model.title_encoder.title_embedding = model.title_encoder.title_embedding.to(device)
    model.to(device)
    
    print("Worker %d is working ... " % rank)
    # Fast check the validation process
    if (cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0):
        validate(cfg, -1, model, device, fast_dev=True)
    
    # Training and validation
    for epoch in range(cfg.training.epochs):
        train(cfg, epoch, rank, model, train_data_loader,
              optimizer, steps_one_epoch, device)
    
        if (cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0):
            validate(cfg, epoch, model, device, subsample=cfg.training.subsample_validation)
            save_checkpoint_by_epoch(model.state_dict(), epoch, cfg.training.model_save_path)
    
    print("Done training")
    if cfg.training.subsample_validation:
        print("Validating on entire set")
        validate(cfg, epoch, model, device, subsample=False)


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def train(cfg, epoch, rank, model, loader, optimizer, steps_one_epoch, device):
    """
    train loop
    :param args: config
    :param epoch: int, the epoch number
    :param gpu_id: int, the gpu id
    :param rank: int, the process rank, equal to gpu_id in this code.
    :param model: gating_model.Model
    :param loader: train data loader.
    :param criterion: loss function
    :param optimizer:
    :param steps_one_epoch: the number of iterations in one epoch
    :return:
    """
    model.train()

    model.zero_grad()

    # enum_dataloader = enumerate(loader)
    # if ((cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0)):
    #     enum_dataloader = enumerate(tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch)))

    print(f"Epoch {epoch}: Training")

    enum_dataloader = enumerate(loader)
    for i, data in enum_dataloader:
        if i >= steps_one_epoch * cfg.training.accumulate:
            break
        data = {key: value.to(device) for key, value in data.items()}
        # 1. Forward
        loss = model.training_step(data)

        if cfg.training.accumulate > 1:
            loss = loss / cfg.training.accumulate

        # 3.Backward.
        loss.backward()

        if ((cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0)) and ((i+1) % cfg.logger.log_freq == 0):
            # neptune.log_metric("loss", loss.item())
            print("loss", loss.item())

        if (i + 1) % cfg.training.accumulate == 0:
            if cfg.training.gpus > 1:
                average_gradients(model)
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            # scheduler.step()
            model.zero_grad()

    # if (not args.dist_train) or (args.dist_train and rank == 0):
    #     util.save_checkpoint_by_epoch(
    #         model.state_dict(), epoch, args.checkpoint_path)


def validate(cfg, epoch, model, device, fast_dev=False, subsample=False):
    model.eval()

    print(f"Epoch {epoch}: Validating")

    valid_dataset = ValidationDataset(cfg.dataset, cfg.dataset.valid)

    # Group by impression IDs and create a mapping from impression IDs to indices
    impression_id_to_indices = {}
    for idx in range(len(valid_dataset)):
        line = valid_dataset.lines[idx]
        impression_id = json.loads(line)["imp_id"]
        if impression_id not in impression_id_to_indices:
            impression_id_to_indices[impression_id] = []
        impression_id_to_indices[impression_id].append(idx)

    # Subsample based on impression IDs if required
    if subsample:
        all_impression_ids = list(impression_id_to_indices.keys())
        subsample_size = min(cfg.training.validation_subsample_size, len(all_impression_ids))
        sampled_impression_ids = random.sample(all_impression_ids, subsample_size)
        
        # Get all indices corresponding to the sampled impression IDs
        subsample_indices = [idx for imp_id in sampled_impression_ids for idx in impression_id_to_indices[imp_id]]
        valid_dataset = Subset(valid_dataset, subsample_indices)
    
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # # Setting the tqdm progress bar
    # data_iter = tqdm(enumerate(valid_data_loader),
    #                  desc="EP_test:%d" % epoch,
    #                  total=len(valid_data_loader),
    #                  bar_format="{l_bar}{r_bar}")

    data_iter = enumerate(valid_data_loader)
    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:
            if fast_dev and i > 10:
                break

            imp_ids += data["imp_id"].cpu().numpy().tolist()
            data = {key: value.to(device) for key, value in data.items()}

            # 1. Forward
            pred = model.prediction_step(data)

            preds += pred.cpu().numpy().tolist()
            truths += data["y"].long().cpu().numpy().tolist()

        all_labels, all_preds = group_labels(truths, preds, imp_ids)
        metric_list = [x.strip() for x in cfg.training.metrics.split("||")]
        ret = cal_metric(all_labels, all_preds, metric_list)
        for metric, val in ret.items():
            print("Epoch: {}, {}: {}".format(epoch, metric, val))


def init_processes(cfg, local_rank, dataset, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    addr = "localhost"
    port = find_free_port()
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=0 + local_rank,
                            world_size=cfg.training.gpus)

    device = torch.device("cuda:{}".format(local_rank))

    fn(cfg, local_rank, device, corpus_path=dataset)


def split_dataset(dataset, gpu_count):
    sub_len = len(dataset) // gpu_count
    if len(dataset) != sub_len * gpu_count:
        len_a, len_b = sub_len * gpu_count, len(dataset) - sub_len * gpu_count
        dataset, _ = torch.utils.data.random_split(dataset, [len_a, len_b])

    return torch.utils.data.random_split(dataset, [sub_len, ] * gpu_count)


def init_exp(cfg):
    if not os.path.exists(cfg.training.model_save_path):
        os.mkdir(cfg.training.model_save_path)


@hydra.main(version_base="1.1", config_path="../conf", config_name="train.yaml")
def main(cfg: DictConfig):
    # if cfg.training.use_doc_embeddings:
    # new_model_save_path = os.path.join(cfg.dataset.result_path, "new_model_save")
    # cfg.training.model_save_path = new_model

    # Determine the suffix based on the flags
    suffix = ""
    if cfg.dataset.sort_one_hop_by_read_time and cfg.dataset.rank_two_hop_by_common_clicks:
        suffix = "_rt_ranked"
    elif cfg.dataset.sort_one_hop_by_read_time:
        suffix = "_rt"
    elif cfg.dataset.rank_two_hop_by_common_clicks:
        suffix = "_ranked"
    
    if cfg.training.use_img_embeddings == True:
        suffix += "_img"
    
    # Update model_save_path with the determined suffix
    cfg.training.model_save_path = os.path.join(cfg.training.model_save_path, f"model{suffix}")
    
    if not os.path.exists(cfg.training.model_save_path):
        os.makedirs(cfg.training.model_save_path)
        print(f"Created directory: {cfg.training.model_save_path}")

    # Print Hydra config args
    print("Hydra Configuration Arguments:")
    print(OmegaConf.to_yaml(cfg))

    # Set the seed for reproducibility
    set_seed(cfg.training.seed)

    if cfg.training.gpus == 0:
        print("== CPU Mode ==")
        datasets = cfg.dataset.train
        run(cfg, 0, torch.device("cpu"), datasets)
    elif cfg.training.gpus == 1:
        datasets = cfg.dataset.train
        run(cfg, 0, torch.device("cuda:0"), datasets)
    else:
        processes = []
        for rank in range(cfg.training.gpus):
            datasets = cfg.dataset.train + ".p{}.tsv".format(rank)
            p = mp.Process(target=init_processes, args=(
                cfg, rank, datasets, run, "nccl"))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # parser = argparse.ArgumentParser()

    # parser.add_argument("--pt_doc_kind", default="word2vec", type=str,
    #                     help="Pt doc emb kind.")
    # parser.add_argument("--use_doc_embeddings", action='store_true', help="Whether to use pretrained document embeddings")

    # global args
    # args, unknown = parser.parse_known_args()

    main()

# /home/scur1584/.conda/envs/recsys/bin/python /home/scur1584/GERL/src/train.py model.name="word_emb" training.epochs=20 dataset.size="ebnerd_demo" training.use_doc_embeddings=False training.subsample_validation=False dataset.sort_one_hop_by_read_time=True dataset.rank_two_hop_by_common_clicks=True