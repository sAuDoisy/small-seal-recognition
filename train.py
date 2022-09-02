from data import DatasetFromObj
from torch.utils.data import DataLoader
from model import S2SModel
import os
import sys
import argparse
import torch
import random
import time
import math

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--Ldiscriminator_penalty', type=int, default=20,
                    help='weight for make discriminator more powerful, g_loss too large, and d_loss too small')
parser.add_argument('--epoch', type=int, default=500, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', type=int, default=20, help='number of epochs to half learning rate')
# in tf, there has default val.
parser.add_argument('--freeze_encoder', action='store_true',
                    help="freeze encoder weights during training")
parser.add_argument('--fine_tune', type=str, default=None,
                    help='specific labels id to be fine tuned')
parser.add_argument('--inst_norm', action='store_true',
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', type=int, default=50,
                    help='number of batches in between two checkpoints')
parser.add_argument('--flip_labels', action='store_true',
                    help='whether flip training data labels or not, in fine tuning')
# to record and get the same random num
parser.add_argument('--random_seed', type=int, default=777,
                    help='random seed for random and pytorch')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--input_nc', type=int, default=3, help='number of input images channels')
# should be 408, because there is no 1
parser.add_argument('--tgt_vocab_size', type=int, default=409,
                    help='number of all radical structures + pad(0) + start(2) + end(3)')


def chkormakedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    # set the seed to get the same random number per time. To get the stable result.
    args = parser.parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    chkormakedir(checkpoint_dir)
    sample_dir = os.path.join(args.experiment_dir, "sample")
    chkormakedir(sample_dir)
    # for i in range(4):
    #     chkormakedir(os.path.join(sample_dir, str(i)))

    start_time = time.time()

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'),
    #                                augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = S2SModel(
        input_nc=args.input_nc,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        tgt_vocab_size=args.tgt_vocab_size
    )
    model.setup()
    model.print_networks(True)
    if args.resume:
        model.load_networks(args.resume)

    # val dataset load only once, no shuffle
    val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'), input_nc=args.input_nc)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    global_steps = 0
    cnt = 0
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch):
        # generate train dataset every epoch so that different styles of saved char imgs can be trained.
        train_dataset = DatasetFromObj(
            os.path.join(data_dir, 'train.obj'),
            input_nc=args.input_nc,
            augment=True,
            bold=False,
            rotate=False,
            blur=True,
        )
        total_batches = math.ceil(len(train_dataset) / args.batch_size)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        for bid, batch in enumerate(dataloader):
            model.set_input(batch[0], batch[2], batch[1], batch[3])
            # label_type = int(batch[0].split('_')[1])

            const_loss, l1_loss, cheat_loss = model.optimize_parameters()
            if bid % 100 == 0:
                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.2f, d_loss: %.5f, g_loss: %.5f, " + \
                             "cheat_loss: %.5f, const_loss: %.5f, l1_loss: %.5f, trm_loss: %.5f"
                print(log_format % (epoch, bid, total_batches, passed, model.d_loss.item(), model.g_loss.item(),
                                    cheat_loss, const_loss, l1_loss, model.trm_pre_loss.item()))
            if global_steps % args.checkpoint_steps == 0:
                model.save_networks(global_steps)
                print("Checkpoint: save checkpoint step %d" % global_steps)
            if global_steps % args.sample_steps == 0:
                for vbid, val_batch in enumerate(val_dataloader):
                    model.sample(val_batch, os.path.join(sample_dir, str(global_steps)))
                    cnt += 1
                print("Sample: sample step %d" % global_steps)
            global_steps += 1
        if (epoch + 1) % args.schedule == 0:
            model.update_lr()
    for vbid, val_batch in enumerate(val_dataloader):
        model.sample(val_batch, os.path.join(sample_dir, str(global_steps)))
        print("Checkpoint: save checkpoint step %d" % global_steps)
        cnt += 1
    model.save_networks(global_steps)


if __name__ == '__main__':
    main()
