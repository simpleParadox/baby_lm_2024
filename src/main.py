import sys
sys.path.append('/home/rsaha/projects/babylm/')
sys.path.append('/home/rsaha/projects/babylm/src/')

from relation_train_net import CustomDetectionModel


import argparse
import torch

parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--skip-test",
    dest="skip_test",
    help="Do not test the final model",
    action="store_true",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

output_dir = cfg.OUTPUT_DIR
if output_dir:
    mkdir(output_dir)

logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
logger.info("Using {} GPUs".format(num_gpus))
logger.info(args)

logger.info("Collecting env info (might take some time)")
logger.info("\n" + collect_env_info())

logger.info("Loaded configuration file {}".format(args.config_file))
with open(args.config_file, "r") as cf:
    config_str = "\n" + cf.read()
    logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
logger.info("Saving config into: {}".format(output_config_path))
# save overloaded model config in the output directory
save_config(cfg, output_config_path)

model = train(cfg, args.local_rank, args.distributed, logger)

if not args.skip_test:
    run_test(cfg, model, args.distributed, logger)








    def main():
        parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
        parser.add_argument(
            "--config-file",
            default="",
            metavar="FILE",
            help="path to config file",
            type=str,
        )
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument(
            "--skip-test",
            dest="skip_test",
            help="Do not test the final model",
            action="store_true",
        )
        parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )

        args = parser.parse_args()

        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        args.distributed = num_gpus > 1

        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()

        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

        output_dir = cfg.OUTPUT_DIR
        if output_dir:
            mkdir(output_dir)

        logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(args)

        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())

        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
        logger.info("Saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)

        model = train(cfg, args.local_rank, args.distributed, logger)

        if not args.skip_test:
            run_test(cfg, model, args.distributed, logger)

