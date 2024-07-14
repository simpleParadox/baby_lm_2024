# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime

import torch
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random

import wandb
from tqdm import tqdm


from apex import amp


# from maskrcnn_benchmark.config import cfg
from detectron2.config import get_cfg

# from maskrcnn_benchmark.data import make_data_loader
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper


# from maskrcnn_benchmark.solver import make_lr_scheduler

from detectron2.solver import build_lr_scheduler

# from maskrcnn_benchmark.solver import make_opimizer
from detectron2.solver import build_optimizer  # After looking at the source code, it does seem like gradient norm clipping is implemented inside the build_optimizer function.
# See this link for more: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/solver/build.html?highlight=gradient%20norm#

# from maskrcnn_benchmark.engine.trainer import reduce_loss_dict # Not used / needed for detectron2.

# from maskrcnn_benchmark.engine.inference import inference

# from maskrcnn_benchmark.modeling.detector import build_detection_model
from detectron2.modeling import build_model


# from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from detectron2.checkpoint import DetectionCheckpointer


from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm


# from maskrcnn_benchmark.utils.collect_env import collect_env_info
from detectron2.utils.collect_env import collect_env_info

# from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from detectron2.utils.comm import synchronize, get_rank, all_gather

# from maskrcnn_benchmark.utils.imports import import_file
from detectron2.utils.imports import import_file





class CustomDetectionModel:
    def __init__(self, cfg):
        self.model = build_model(cfg)
        self.distributed = False
        random_seed = 42  # TODO: Might need to do this change this to across 5 seeds.
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False 
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)



    def train(self, cfg, local_rank, distributed, logger):
        model = build_model(cfg) 

        # the modules that should be always set in eval mode
        # their eval() method should be called after model.train() is called
        if cfg.WSVL.OFFLINE_OD:  # offline object detector
            eval_modules = (model.roi_heads.box,)
        else:  # online object detector
            eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
        self.fix_eval_modules(eval_modules)

        # NOTE, we slow down the LR of the layers start with the names in slow_heads
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
            slow_heads = ["roi_heads.relation.box_feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor",]
        else:
            slow_heads = []

        # load pretrain layers to new layers
        load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
        
        if cfg.MODEL.ATTRIBUTE_ON:
            load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
            load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        num_batch = cfg.SOLVER.IMS_PER_BATCH
        optimizer = build_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
        scheduler = build_lr_scheduler(cfg, optimizer, logger)
        # debug_print(logger, 'end optimizer and shcedule')

        # Initialize mixed-precision training
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        amp_opt_level = 'O2' if cfg.WSVL.USE_UNITER and use_mixed_precision else amp_opt_level # in 'o1' mode, FusedLayerNorm can't accept float16 = nn.Linear(float32)
        amp_opt_level = 'O1' if not cfg.WSVL.OFFLINE_OD and use_mixed_precision else amp_opt_level # online detector has to use 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level) #, patch_torch_functions=False)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        # debug_print(logger, 'end distributed')

        arguments = {}
        arguments["iteration"] = 0
        output_dir = cfg.OUTPUT_DIR
        save_to_disk = get_rank() == 0
        checkpointer = DetectionCheckpointer(
            model, output_dir, optimizer=optimizer, scheduler=scheduler, save_to_disk=save_to_disk
        )
        # if there is file 'last_checkpoint' (contains the name of last ckpt) in output_dir, load it, else load pretrained detector
        if checkpointer.has_checkpoint():
            extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, 
                                        update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
            arguments.update(extra_checkpoint_data)            
        else:
            if cfg.WSVL.OFFLINE_OD:  # offline object detector
                # set cfg.MODEL.PRETRAINED_DETECTOR_CKPT as its default value ''
                checkpointer.load('', with_optim=False, load_mapping=load_mapping)
            else:  # online object detector
                # load_mapping is only used when we init current model from detection model.
                checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        # debug_print(logger, 'end load checkpointer')

        train_data_loader = build_detection_train_loader(
            cfg,
            mode='train',
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        val_data_loaders = build_detection_test_loader(
            cfg,
            mode='val',
            is_distributed=distributed,
        )
        # debug_print(logger, 'end dataloader')

        if cfg.SOLVER.PRE_VAL:
            logger.info("Validate before training")
            self.run_val(cfg, model, val_data_loaders, distributed, logger)

        if cfg.WSVL.SKIP_TRAIN:  # if skip train, then evaluate the loaded model / initialized model directly
            return model

        logger.info("Start training")
        # meters = MetricLogger(delimiter="  ")  # TODO: replace with wandb.
        max_iter = len(train_data_loader)
        start_iter = arguments["iteration"]
        start_training_time = time.time()
        end = time.time()
        print_first_grad = True
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        for iteration, this_batch in enumerate(train_data_loader, start_iter):
            if cfg.WSVL.OFFLINE_OD:  # offline object detector detection results
                if cfg.WSVL.USE_UNITER:
                    det_feats, det_dists, det_boxes, targets, _, images, det_tag_ids, det_norm_pos = this_batch
                else:
                    det_feats, det_dists, det_boxes, targets, _, images = this_batch
            else:  # online object detector
                images, targets, _ = this_batch
            
            if any(len(target) < 1 for target in targets):
                logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            model.train()
            self.fix_eval_modules(eval_modules)

            if cfg.WSVL.OFFLINE_OD:  # offline object detector detection results
                if cfg.WSVL.USE_UNITER:
                    images = images.to(device)
                    det_feats = [det_feat.to(device) for det_feat in det_feats]
                    det_dists = [det_dist.to(device) for det_dist in det_dists]
                    det_boxes = [det_box.to(device) for det_box in det_boxes]
                    targets = [target.to(device) for target in targets]
                    det_tag_ids = [[det_tag_i.to(device) for det_tag_i in det_tag_id] for det_tag_id in det_tag_ids]
                    det_norm_pos = [det_norm_p.to(device) for det_norm_p in det_norm_pos]
                    loss_dict = model(images, targets, det_feats=det_feats, det_dists=det_dists, det_boxes=det_boxes,\
                                    det_tag_ids=det_tag_ids, det_norm_pos=det_norm_pos)
                else:
                    images = images.to(device)
                    det_feats = [det_feat.to(device) for det_feat in det_feats]
                    det_dists = [det_dist.to(device) for det_dist in det_dists]
                    det_boxes = [det_box.to(device) for det_box in det_boxes]
                    targets = [target.to(device) for target in targets]
                    loss_dict = model(images, targets, det_feats=det_feats, det_dists=det_dists, det_boxes=det_boxes)  
            else: # online object detector
                images = images.to(device)
                targets = [target.to(device) for target in targets]
                loss_dict = model(images, targets)

            # reduce losses over all GPUs for logging purposes   # Need to change this to be run on single gpu for now.
            losses = sum(loss for loss in loss_dict.values()) # Will contain single value so the sum is the value itself. Currently undefined for multi-gpu training.




            # loss_dict_reduced = reduce_loss_dict(loss_dict)
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # TODO: replace with wandb.
            # meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:  # For mixed precision training.
                scaled_losses.backward()
            
            # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
            verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
            print_first_grad = False
            clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            # TODO: replace with wandb.
            # meters.update(time=batch_time, data=data_time)


            # TODO: Use tqdm here.
            # eta_seconds = meters.time.global_avg * (max_iter - iteration)
            # eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))



            # TODO: replace with wandb.
            # if iteration % 20 == 0 or iteration == max_iter:
            #     logger.info(
            #         meters.delimiter.join(
            #             [
            #                 "eta: {eta}",
            #                 "iter: {iter}",
            #                 "{meters}",
            #                 "lr: {lr:.6f}",
            #                 "max mem: {memory:.0f}",
            #             ]
            #         ).format(
            #             eta=eta_string,
            #             iter=iteration,
            #             meters=str(meters),
            #             lr=optimizer.param_groups[-1]["lr"],
            #             memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            #         )
            #     )

            if iteration % checkpoint_period == 0: # TODO: May need to change this to saving the best model.
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

            val_result = None # used for scheduler updating
            if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
                # logger.info("Start validating")
                val_result = self.run_val(cfg, model, val_data_loaders, distributed, logger)
                # logger.info("Validation Result: %.4f" % val_result)
    
            # scheduler should be called after optimizer.step() in pytorch>=1.1.0
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
                scheduler.step(val_result, epoch=iteration)
                if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                    # logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                    break
            else:
                scheduler.step()

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )
        return model





    def fix_eval_modules(self, eval_modules):
        for module in eval_modules:
            for _, param in module.named_parameters():
                param.requires_grad = False



    def run_test(self, cfg, model, distributed, logger):
        if distributed:
            model = model.module
        torch.cuda.empty_cache()
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        if cfg.MODEL.RELATION_ON:
            iou_types = iou_types + ("relations", )
        if cfg.MODEL.ATTRIBUTE_ON:
            iou_types = iou_types + ("attributes", )
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                os.makedirs(output_folder, exist_ok=True)
                output_folders[idx] = output_folder
        data_loaders_val = build_detection_test_loader(cfg, mode='test', is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            self.inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                logger=logger,
                offline_od=cfg.WSVL.OFFLINE_OD,
                use_uniter=cfg.WSVL.USE_UNITER
            )
            synchronize()


    def run_val(self, cfg, model, val_data_loaders, distributed, logger):
        if distributed:
            model = model.module
        torch.cuda.empty_cache()
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        if cfg.MODEL.RELATION_ON:
            iou_types = iou_types + ("relations", )
        if cfg.MODEL.ATTRIBUTE_ON:
            iou_types = iou_types + ("attributes", )

        dataset_names = cfg.DATASETS.VAL
        val_result = []
        for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
            dataset_result = self.inference(
                                cfg,
                                model,
                                val_data_loader,
                                dataset_name=dataset_name,
                                iou_types=iou_types,
                                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                                device=cfg.MODEL.DEVICE,
                                expected_results=cfg.TEST.EXPECTED_RESULTS,
                                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                                output_folder=None,
                                logger=logger,
                                offline_od=cfg.WSVL.OFFLINE_OD,
                                use_uniter=cfg.WSVL.USE_UNITER
                            )
            synchronize()
            val_result.append(dataset_result)
        # support for multi gpu distributed testing
        gathered_result = all_gather(torch.tensor(dataset_result).cpu())
        gathered_result = [t.view(-1) for t in gathered_result]
        gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
        valid_result = gathered_result[gathered_result>=0]
        val_result = float(valid_result.mean())
        del gathered_result, valid_result
        torch.cuda.empty_cache()
        return val_result
    

    def compute_on_dataset(self, model, data_loader, device, bbox_aug, timer=None):
        model.eval()
        results_dict = {}
        cpu_device = torch.device("cpu")
        for _, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch
            with torch.no_grad():
                # if timer:
                #     timer.tic()
                # if bbox_aug:
                #     output = im_detect_bbox_aug(model, images, device)
                # else:
                output = model(images.to(device))
                # if timer:
                #     if not device.type == 'cpu':
                #         torch.cuda.synchronize()
                #     timer.toc()
                output = [o.to(cpu_device) for o in output]
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        return results_dict
    


    def inference(self, model, data_loader, dataset_name, iou_types=("bbox",), box_only=False, bbox_aug=False, device="cuda", expected_results=(),
        expected_results_sigma_tol=4, output_folder=None, ):
        
        # convert to a torch.device for efficiency
        device = torch.device(device)
        num_devices = self.get_world_size()
        # logger = logging.getLogger("maskrcnn_benchmark.inference")
        dataset = data_loader.dataset
        # logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
        # total_timer = Timer()
        # inference_timer = Timer()
        # total_timer.tic()
        predictions = self.compute_on_dataset(model, data_loader, device, bbox_aug)
        # wait for all processes to complete before measuring the time
        synchronize()

        # TODO: Replace with tqdm and wandb as necessary.
        
        # total_time = total_timer.toc()
        # total_time_str = get_time_str(total_time)
        # logger.info(
        #     "Total run time: {} ({} s / img per device, on {} devices)".format(
        #         total_time_str, total_time * num_devices / len(dataset), num_devices
        #     )
        # )
        # total_infer_time = get_time_str(inference_timer.total_time)
        # logger.info(
        #     "Model inference time: {} ({} s / img per device, on {} devices)".format(
        #         total_infer_time,
        #         inference_timer.total_time * num_devices / len(dataset),
        #         num_devices,
        #     )
        # )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        if not is_main_process():
            return

        if output_folder:
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

        extra_args = dict(
            box_only=box_only,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )

        return self.evaluate(dataset=dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)
    

    def evaluate(self, dataset, predictions, output_folder, **kwargs):
        """evaluate dataset using different methods based on dataset type.
        Args:
            dataset: Dataset object
            predictions(list[BoxList]): each item in the list represents the
                prediction results for one image.
            output_folder: output folder, to save evaluation files or results.
            **kwargs: other args.
        Returns:
            evaluation result
        """
        args = dict(
            dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
        )
        if isinstance(dataset, datasets.COCODataset):
            return coco_evaluation(**args)
        elif isinstance(dataset, datasets.PascalVOCDataset):
            return voc_evaluation(**args)
        elif isinstance(dataset, datasets.AbstractDataset):
            return abs_cityscapes_evaluation(**args)
        else:
            dataset_name = dataset.__class__.__name__
            raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
        

    def get_world_size(self):
        return 1