import os
import time
import torch
import logging
import contextlib
import numpy as np
import torch.nn as nn
import detectron2.utils.comm as comm

from module import FENs
from utils import LossEvalHook, save_config
from data.mapper import generate_mapper_using_cfg

from collections import OrderedDict
from detectron2.engine.train_loop import TrainerBase
from detectron2.config import CfgNode
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    COCOEvaluator
)
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, hooks
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter


try:
    _nullcontext = contextlib.nullcontext  # python 3.7+
except AttributeError:

    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result


class SimpleTrainer(TrainerBase):
    def __init__(self, model, optimizer, data_loader, fen=None, fen_optimizer=None):

        super().__init__()
        model.train()

        self.model = model
        self.optimizer = optimizer

        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

        self.fen = fen
        self.fen_optimizer = fen_optimizer

    def run_step(self):
        assert self.model.training, \
            "Model is not in train mode."

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # Update Object Detector
        self.optimizer.zero_grad()
        loss_dict, cloned_features = self.model(data)
        losses = sum(loss_dict.values())
        losses.backward()
        self.optimizer.step()

        # Update FEN if FEN is activated.
        if (self.fen is not None) and (self.fen_optimizer is not None):
            self.fen_optimizer.zero_grad()
            fen_loss = self.fen.get_loss(cloned_features)
            fen_loss.backward()
            self.fen_optimizer.step()

            # Transfer Updated Weight
            for (s_m, d_m) in zip(self.fen.module_list, self.model.denoiser.module_list):
                if not (isinstance(s_m, nn.Identity) and isinstance(d_m, nn.Identity)):
                    self.transfer_weight(s_m, d_m)

        # Update Losses
        with torch.cuda.stream(torch.cuda.Stream()) \
                if losses.device.type == "cuda" else _nullcontext():

            metrics_dict = {}
            metrics_dict.update(loss_dict)
            if (self.fen is not None) and (self.fen_optimizer is not None):
                metrics_dict.update({'fen_loss': fen_loss})
            metrics_dict.update({'data_time': data_time})

            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}! Loss dict is {}.\n".format(self.iter, loss_dict))

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    def transfer_weight(self, src_model: nn.Module, dst_model: nn.Module):
        src_dict = src_model.state_dict()
        dst_dict = dst_model.state_dict()
        assert len(src_dict.keys()) == len(dst_dict.keys()), \
            "`src_model` and `dst_model` seems different."

        for src_key, dst_key in zip(src_dict.keys(), dst_dict.keys()):
            dst_dict[dst_key] = src_dict[src_key]
        dst_model.load_state_dict(dst_dict)


class DefaultTrainer(SimpleTrainer):
    def __init__(self, cfg: CfgNode):

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self.build_model(cfg).to(device)
        optimizer = self.build_main_optimizer(cfg, model)

        fen = None
        fen_optimizer = None

        if cfg.MODEL.FEN.USE_FEN:
            fen = FENs(cfg, init_freeze=False).to(device)
            fen_optimizer = self.build_fen_optimizer(fen, lr=2.5e-04)

        data_loader = self.build_train_loader(cfg)

        super().__init__(model=model,
                         fen=fen,
                         optimizer=optimizer,
                         fen_optimizer=fen_optimizer,
                         data_loader=data_loader)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [hooks.IterationTimer(), hooks.LRScheduler(self.optimizer, self.scheduler)]
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.LOG_PERIOD))
        return ret

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        super().train(self.start_iter, self.max_iter)

    def build_train_loader(self, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=generate_mapper_using_cfg(cfg))

    def build_test_loader(self, cfg: CfgNode, dataset_name: str):
        return build_detection_test_loader(cfg, dataset_name, mapper=generate_mapper_using_cfg(cfg))

    def build_model(self, cfg):
        model = build_model(cfg)
        return model

    def build_main_optimizer(self, cfg, model):
        return build_optimizer(cfg, model)

    def build_fen_optimizer(self, model, lr: float = 2.5e-04):
        print(model)
        return torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))

    def build_lr_scheduler(self, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def build_evaluator(self, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def test(self, cfg, model, evaluators=None):
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = self.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = self.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


class FENTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        mapper = generate_mapper_using_cfg(self.cfg)
        hooks.insert(-1,
                     LossEvalHook(self.cfg.TEST.EVAL_PERIOD,
                                  self.model,
                                  build_detection_test_loader(self.cfg,
                                                              self.cfg.DATASETS.TEST[0],
                                                              mapper)))
        return hooks
