"""
Resume training script for AASIST from a checkpoint.
Modified from main.py to support resuming from epoch 44 to 100.
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / dev / eval"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    d_label_trn, file_train = genSpoof_list(
        dir_meta=Path(database_path) / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.train.trn.txt",
        is_train=True,
        is_eval=False)

    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=Path(database_path) / f"ASVspoof2019_{track}_train/")

    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen,
                            num_workers=0)

    _, file_dev = genSpoof_list(
        dir_meta=Path(database_path) / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.dev.trl.txt",
        is_train=False,
        is_eval=False)

    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=Path(database_path) / f"ASVspoof2019_{track}_dev/")
    dev_loader = DataLoader(dev_set,
                           batch_size=config["batch_size"],
                           shuffle=False,
                           drop_last=False,
                           pin_memory=True,
                           num_workers=0)

    file_eval = genSpoof_list(
        dir_meta=Path(database_path) / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.eval.trl.txt",
        is_train=False,
        is_eval=True)

    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=Path(database_path) / f"ASVspoof2019_{track}_eval/")
    eval_loader = DataLoader(eval_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=0)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Produce output scores for evaluation"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optimizer,
    device: torch.device,
    scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()

    running_loss /= num_total
    return running_loss


def main(args: argparse.Namespace) -> None:
    """Main function for resuming training"""
    # Load configuration
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # Set seed
    set_seed(args.seed, config)

    # Set output directory
    model_tag = "{}_{}_ep{}_bs{}".format(
        track, model_config["architecture"],
        config["num_epochs"], config["batch_size"])

    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = Path(args.output_dir) / model_tag

    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                     f"ASVspoof2019_{track}_cm_protocols/ASVspoof2019.{track}.cm.dev.trl.txt")
    eval_trial_path = (database_path /
                      f"ASVspoof2019_{track}_cm_protocols/ASVspoof2019.{track}.cm.eval.trl.txt")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # Define model
    model = get_model(model_config, device)

    # Load checkpoint
    checkpoint_path = args.resume_checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Checkpoint loaded successfully!")

    # Get dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # Get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # Prepare for resuming
    start_epoch = args.start_epoch
    best_dev_eer = args.best_dev_eer if args.best_dev_eer is not None else 1.0
    best_eval_eer = 100.0
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.0
    n_swa_update = 0

    # Setup paths
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / "eval_scores_using_best_dev_model.txt"
    writer = SummaryWriter(model_tag)

    # Metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write(f"\n===== RESUMED FROM EPOCH {start_epoch} =====\n")

    # Training loop
    print(f"\nResuming training from epoch {start_epoch} to {config['num_epochs']}")
    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"Start training epoch{epoch:03d}")
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)

        # Validation
        produce_evaluation_file(dev_loader, model, device,
                               metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/f"dev_t-DCF_EER_{epoch:03d}epo.txt",
            printout=False)

        print(f"DONE.\nLoss:{running_loss:.5f}, dev_eer: {dev_eer:.3f}, dev_tdcf:{dev_tdcf:.5f}")
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        # Save log
        f_log.write(f"epoch:{epoch}, loss:{running_loss:.5f}, dev_eer:{dev_eer:.5f}, dev_tdcf:{dev_tdcf:.5f}\n")
        f_log.flush()

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print(f"Best model found at epoch {epoch}")
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                      model_save_path / f"epoch_{epoch}_{dev_eer:03.3f}.pth")
            torch.save(model.state_dict(), model_save_path / "best.pth")

            # Evaluate on test set
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device,
                                       eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path/config["asv_score_path"],
                    output_file=metric_path/f"t-DCF_EER_{epoch:03d}epo.txt")

                print(f"DONE.\nEval EER: {eval_eer:.3f}, Eval t-DCF: {eval_tdcf:.5f}")
                writer.add_scalar("eval_eer", eval_eer, epoch)
                writer.add_scalar("eval_tdcf", eval_tdcf, epoch)

                best_eval_eer = min(eval_eer, best_eval_eer)
                best_eval_tdcf = min(eval_tdcf, best_eval_tdcf)

        # SWA
        if epoch >= config["num_epochs"] - 10:
            optimizer_swa.update_swa()
            n_swa_update += 1
            print(f"Saving epoch {epoch} for SWA")

    # Final summary
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best dev EER: {best_dev_eer:.3f}")
    print(f"Best eval EER: {best_eval_eer:.3f}")
    print(f"{'='*50}\n")

    f_log.write(f"\n===== TRAINING COMPLETED =====\n")
    f_log.write(f"Best dev EER: {best_dev_eer:.5f}\n")
    f_log.write(f"Best eval EER: {best_eval_eer:.5f}\n")
    f_log.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume AASIST training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--output_dir", type=str, default="exp_result", help="Output directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--resume_checkpoint", type=str, required=True, help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, required=True, help="Epoch to start from")
    parser.add_argument("--best_dev_eer", type=float, default=None, help="Best dev EER so far")
    parser.add_argument("--comment", type=str, default=None, help="Comment for model tag")

    args = parser.parse_args()
    main(args)
