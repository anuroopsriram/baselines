import datetime
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ocpmodels.common import distutils
from ocpmodels.common.display import Display
from ocpmodels.common.logger import TensorboardLogger, WandBLogger
from ocpmodels.common.meter import Meter, mae, mae_ratio, mean_l2_distance
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    plot_histogram,
    save_checkpoint,
    update_config,
    warmup_lr_lambda,
)
from ocpmodels.datasets import (
    ISO17,
    Gasdb,
    QM9Dataset,
    UlissigroupCO,
    UlissigroupH,
    XieGrossmanMatProj,
)
from ocpmodels.models import CGCNN, CGCNNGu, Transformer
from ocpmodels.modules.normalizer import Normalizer


@registry.register_trainer("base")
class BaseTrainer:
    def __init__(self, args=None, local_rank=0):
        # defaults.
        self.device = "cpu"
        self.is_debug = True
        self.is_vis = True

        # load config.
        if args is not None:
            self.load_config_from_yaml_and_cmd(args)

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_task()
        self.load_model()
        self.load_criterion()
        self.load_optimizer()
        self.load_extras()

    # Note: this function is now deprecated. We build config outside of trainer.
    # See build_config in ocpmodels.common.utils.py.
    def load_config_from_yaml_and_cmd(self, args):
        self.config = build_config(args)

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config["amp"] else None

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Are we just running sanity checks?
        self.is_debug = args.debug
        self.is_vis = args.vis

        # timestamps and directories
        args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if args.identifier:
            args.timestamp += "-{}".format(args.identifier)

        args.checkpoint_dir = os.path.join("checkpoints", args.timestamp)
        args.results_dir = os.path.join("results", args.timestamp)
        args.logs_dir = os.path.join(
            "logs", self.config["logger"], args.timestamp
        )

        print(yaml.dump(self.config, default_flow_style=False))
        for arg in vars(args):
            print("{:<20}: {}".format(arg, getattr(args, arg)))

        # TODO(abhshkdz): Handle these parameters better. Maybe move to yaml.
        self.config["cmd"] = args.__dict__
        del args

        if not self.is_debug:
            os.makedirs(self.config["cmd"]["checkpoint_dir"])
            os.makedirs(self.config["cmd"]["results_dir"])
            os.makedirs(self.config["cmd"]["logs_dir"])

            # Dump config parameters
            json.dump(
                self.config,
                open(
                    os.path.join(
                        self.config["cmd"]["checkpoint_dir"], "config.json"
                    ),
                    "w",
                ),
            )

    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug and distutils.is_master():
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"
            self.logger = registry.get_logger_class(self.config["logger"])(
                self.config
            )

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))
        dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
            self.config["dataset"]
        )

        if self.config["task"]["dataset"] in ["qm9", "dogss"]:
            num_targets = dataset.data.y.shape[-1]
            if (
                "label_index" in self.config["task"]
                and self.config["task"]["label_index"] is not False
            ):
                dataset.data.y = dataset.data.y[
                    :, int(self.config["task"]["label_index"])
                ]
                num_targets = 1
        else:
            num_targets = 1

        self.num_targets = num_targets
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = dataset.get_dataloaders(
            batch_size=int(self.config["optim"]["batch_size"])
        )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.config["dataset"].get("normalize_labels", True):
            self.normalizers["target"] = Normalizer(
                self.train_loader.dataset.data.y[
                    self.train_loader.dataset.__indices__
                ],
                self.device,
            )

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if "grad_input" in self.config["task"]:
            if self.config["dataset"].get("normalize_labels", True):
                self.normalizers["grad_target"] = Normalizer(
                    self.train_loader.dataset.data.y[
                        self.train_loader.dataset.__indices__
                    ],
                    self.device,
                )
                self.normalizers["grad_target"].mean.fill_(0)

        if self.is_vis and self.config["task"]["dataset"] != "qm9":
            # Plot label distribution.
            plots = [
                plot_histogram(
                    self.train_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: train",
                ),
                plot_histogram(
                    self.val_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: val",
                ),
                plot_histogram(
                    self.test_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: test",
                ),
            ]
            self.logger.log_plots(plots)

    def load_model(self):
        # Build model
        print("### Loading model: {}".format(self.config["model"]))

        # TODO(abhshkdz): Eventually move towards computing features on-the-fly
        # and remove dependence from `.edge_attr`.
        bond_feat_dim = None
        if self.config["task"]["dataset"] in [
            "ulissigroup_co",
            "ulissigroup_h",
            "xie_grossman_mat_proj",
        ]:
            bond_feat_dim = self.train_loader.dataset[0].edge_attr.shape[-1]
        elif self.config["task"]["dataset"] in [
            "gasdb",
            "trajectory",
            "trajectory_lmdb",
            "single_point_lmdb",
        ]:
            bond_feat_dim = self.config["model_attributes"].get(
                "num_gaussians", 50
            )
        else:
            raise NotImplementedError

        self.model = registry.get_model_class(self.config["model"])(
            self.train_loader.dataset[0].x.shape[-1]
            if hasattr(self.train_loader.dataset[0], "x")
            and self.train_loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        print(
            "### Loaded {} with {} parameters.".format(
                self.model.__class__.__name__, self.model.num_params
            )
        )

        if self.logger is not None:
            self.logger.watch(self.model)

    def load_pretrained(self, checkpoint_path=None, ddp_to_dp=False):
        if checkpoint_path is None or os.path.isfile(checkpoint_path) is False:
            return False

        print("### Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        if ddp_to_dp:
            new_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]
                new_dict[name] = v
            self.model.load_state_dict(new_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])
        return True

    # TODO(abhshkdz): Rename function to something nicer.
    # TODO(abhshkdz): Support multiple loss functions.
    def load_criterion(self):
        self.criterion = self.config["optim"].get("criterion", nn.L1Loss())

    def load_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            self.config["optim"]["lr_initial"],  # weight_decay=3.0
        )

    def load_extras(self):
        # learning rate scheduler.
        scheduler_lambda_fn = lambda x: warmup_lr_lambda(
            x, self.config["optim"]
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=scheduler_lambda_fn
        )

        # metrics.
        self.meter = Meter(split="train")

    def train(self, max_epochs=None, return_metrics=False):
        # TODO(abhshkdz): Timers for dataloading and forward pass.
        num_epochs = (
            max_epochs
            if max_epochs is not None
            else self.config["optim"]["max_epochs"]
        )
        for epoch in range(num_epochs):
            self.model.train()

            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)

                # Forward, loss, backward.
                out, metrics = self._forward(batch)
                loss = self._compute_loss(out, batch)
                self._backward(loss)

                # Update meter.
                meter_update_dict = {
                    "epoch": epoch + (i + 1) / len(self.train_loader),
                    "loss": loss.item(),
                }
                meter_update_dict.update(metrics)
                self.meter.update(meter_update_dict)

                # Make plots.
                if self.logger is not None:
                    self.logger.log(
                        meter_update_dict,
                        step=epoch * len(self.train_loader) + i + 1,
                        split="train",
                    )

                # Print metrics.
                if i % self.config["cmd"]["print_every"] == 0:
                    print(self.meter)

            self.scheduler.step()

            with torch.no_grad():
                if self.val_loader is not None:
                    v_loss, v_mae = self.validate(split="val", epoch=epoch)

                if self.test_loader is not None:
                    test_loss, test_mae = self.validate(
                        split="test", epoch=epoch
                    )

            if not self.is_debug:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "amp": self.scaler.state_dict() if self.scaler else None,
                    },
                    self.config["cmd"]["checkpoint_dir"],
                )
        if return_metrics:
            return {
                "training_loss": float(self.meter.loss.global_avg),
                "training_mae": float(
                    self.meter.meters[
                        self.config["task"]["labels"][0]
                        + "/"
                        + self.config["task"]["metric"]
                    ].global_avg
                ),
                "validation_loss": v_loss,
                "validation_mae": v_mae,
                "test_loss": test_loss,
                "test_mae": test_mae,
            }

    def validate(self, split="val", epoch=None):
        print("### Evaluating on {}.".format(split))
        self.model.eval()

        meter = Meter(split=split)

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in enumerate(loader):
            batch = batch.to(self.device)

            # Forward.
            out, metrics = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Update meter.
            meter_update_dict = {"loss": loss.item()}
            meter_update_dict.update(metrics)
            meter.update(meter_update_dict)

        # Make plots.
        if self.logger is not None and epoch is not None:
            log_dict = meter.get_scalar_dict()
            log_dict.update({"epoch": epoch + 1})
            self.logger.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )

        print(meter)
        return (
            float(meter.loss.global_avg),
            float(
                meter.meters[
                    self.config["task"]["labels"][0]
                    + "/"
                    + self.config["task"]["metric"]
                ].global_avg
            ),
        )

    def _forward(self, batch, compute_metrics=True):
        out = {}

        # enable gradient wrt input.
        if "grad_input" in self.config["task"]:
            inp_for_grad = batch.pos
            batch.pos = batch.pos.requires_grad_(True)

        # forward pass.
        if self.config["model_attributes"].get("regress_forces", False):
            output, output_forces = self.model(batch)
        else:
            output = self.model(batch)

        if output.shape[-1] == 1:
            output = output.view(-1)

        out["output"] = output

        force_output = None
        if self.config["model_attributes"].get("regress_forces", False):
            out["force_output"] = output_forces
            force_output = output_forces

        if (
            "grad_input" in self.config["task"]
            and self.config["model_attributes"].get("regress_forces", False)
            is False
        ):
            force_output = (
                -1
                * torch.autograd.grad(
                    output,
                    inp_for_grad,
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
            out["force_output"] = force_output

        if not compute_metrics:
            return out, None

        metrics = {}

        if self.config["dataset"].get("normalize_labels", True):
            errors = eval(self.config["task"]["metric"])(
                self.normalizers["target"].denorm(output).cpu(), batch.y.cpu()
            ).view(-1)
        else:
            errors = eval(self.config["task"]["metric"])(
                output.cpu(), batch.y.cpu()
            ).view(-1)

        if (
            "label_index" in self.config["task"]
            and self.config["task"]["label_index"] is not False
        ):
            # TODO(abhshkdz): Get rid of this edge case for QM9.
            # This is only because QM9 has multiple targets and we can either
            # jointly predict all of them or one particular target.
            metrics[
                "{}/{}".format(
                    self.config["task"]["labels"][
                        self.config["task"]["label_index"]
                    ],
                    self.config["task"]["metric"],
                )
            ] = errors[0]
        else:
            for i, label in enumerate(self.config["task"]["labels"]):
                metrics[
                    "{}/{}".format(label, self.config["task"]["metric"])
                ] = errors[i]

        if "grad_input" in self.config["task"]:
            force_pred = force_output
            force_target = batch.force

            if self.config["task"].get("eval_on_free_atoms", True):
                mask = batch.fixed == 0
                force_pred = force_pred[mask]
                force_target = force_target[mask]

            if self.config["dataset"].get("normalize_labels", True):
                grad_input_errors = eval(self.config["task"]["metric"])(
                    self.normalizers["grad_target"].denorm(force_pred).cpu(),
                    force_target.cpu(),
                )
            else:
                grad_input_errors = eval(self.config["task"]["metric"])(
                    force_pred.cpu(), force_target.cpu()
                )
            metrics[
                "force_x/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[0]
            metrics[
                "force_y/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[1]
            metrics[
                "force_z/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[2]

        return out, metrics

    def _compute_loss(self, out, batch):
        loss = []

        if self.config["dataset"].get("normalize_labels", True):
            target_normed = self.normalizers["target"].norm(batch.y)
        else:
            target_normed = batch.y

        loss.append(self.criterion(out["output"], target_normed))

        # TODO(abhshkdz): Test support for gradients wrt input.
        # TODO(abhshkdz): Make this general; remove dependence on `.forces`.
        if "grad_input" in self.config["task"]:
            if self.config["dataset"].get("normalize_labels", True):
                grad_target_normed = self.normalizers["grad_target"].norm(
                    batch.force
                )
            else:
                grad_target_normed = batch.force

            # Force coefficient = 30 has been working well for us.
            force_mult = self.config["optim"].get("force_coefficient", 30)
            if self.config["task"].get("train_on_free_atoms", False):
                mask = batch.fixed == 0
                loss.append(
                    force_mult
                    * self.criterion(
                        out["force_output"][mask], grad_target_normed[mask]
                    )
                )
            else:
                loss.append(
                    force_mult
                    * self.criterion(out["force_output"], grad_target_normed)
                )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # TODO(abhshkdz): Add support for gradient clipping.
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
