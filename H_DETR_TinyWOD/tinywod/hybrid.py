"""Original engine.train_hybrid mathematics, shared by both entry points.

H-DETR Copyright (c) 2022 Peking University & Microsoft Research Asia.
Original source and license are preserved in this project.
"""
import copy
from contextlib import contextmanager
from functools import wraps


@contextmanager
def one2one_inference(model):
    """Official inference query switch, also safe for CPU, non-DDP and exceptions."""
    base = model.module if hasattr(model, "module") else model
    queries = base.num_queries
    proposals = base.transformer.two_stage_num_proposals
    training = model.training
    try:
        model.eval()
        base.num_queries = base.num_queries_one2one
        base.transformer.two_stage_num_proposals = base.num_queries_one2one
        yield base
    finally:
        base.num_queries = queries
        base.transformer.two_stage_num_proposals = proposals
        model.train(training)


def one2one_evaluation(function):
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        with one2one_inference(model):
            return function(model, *args, **kwargs)
    return wrapped


def train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many):
    loss_dict = criterion(outputs, targets)
    multi_targets = copy.deepcopy(targets)
    for target in multi_targets:
        target["boxes"] = target["boxes"].repeat(k_one2many, 1)
        target["labels"] = target["labels"].repeat(k_one2many)
    outputs_one2many = {
        "pred_logits": outputs["pred_logits_one2many"],
        "pred_boxes": outputs["pred_boxes_one2many"],
        "aux_outputs": outputs["aux_outputs_one2many"],
    }
    loss_dict_one2many = criterion(outputs_one2many, multi_targets)
    for key, value in loss_dict_one2many.items():
        if key + "_one2many" in loss_dict:
            loss_dict[key + "_one2many"] += value * lambda_one2many
        else:
            loss_dict[key + "_one2many"] = value * lambda_one2many
    return loss_dict
