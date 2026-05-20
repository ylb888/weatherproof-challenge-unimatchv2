import argparse
from copy import deepcopy
import logging
import os
import pprint
from pathlib import Path

import distutils.version
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import yaml
from PIL import Image

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--val-id-path', type=str, default=None)
parser.add_argument('--no-val', action='store_true',
                    help='skip validation during training')
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--save-interval', type=int, default=0,
                    help='save an extra checkpoint every N epochs; 0 disables interval checkpoints')
parser.add_argument('--reference-pred-dir', type=str, default=None,
                    help='compare EMA predictions on unlabeled images with this prediction directory after each epoch')
parser.add_argument('--reference-input-dir', type=str, default='data/test_input',
                    help='image root used for reference prediction comparison')
parser.add_argument('--reference-id-path', type=str, default=None,
                    help='image id list used for reference prediction comparison; defaults to unlabeled id path')
parser.add_argument('--reference-resize-multiple', type=int, default=14)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def prediction_mask_root(path):
    path = Path(path)
    if (path / 'mask').is_dir():
        return path / 'mask'
    return path


def load_reference_masks(path):
    root = prediction_mask_root(path)
    masks = {
        mask.relative_to(root): mask
        for mask in root.rglob('*.png')
        if not mask.name.endswith('_color.png')
    }
    if not masks:
        raise FileNotFoundError('No reference masks found under %s' % root)
    return root, masks


def image_rel_from_id(sample_id):
    return sample_id if Path(sample_id).suffix else sample_id + '_degraded.png'


def mask_rel_from_image_rel(image_rel):
    image_rel = Path(image_rel)
    if image_rel.name.endswith('_degraded.png'):
        return image_rel.with_name(image_rel.name.replace('_degraded.png', '_gt-intern.png'))
    return image_rel.with_name(image_rel.stem + '_gt-intern.png')


def resize_to_multiple(image_tensor, multiple):
    if multiple <= 1:
        return image_tensor, image_tensor.shape[-2:]

    ori_h, ori_w = image_tensor.shape[-2:]
    new_h = int(ori_h / multiple + 0.5) * multiple
    new_w = int(ori_w / multiple + 0.5) * multiple
    new_h = max(new_h, multiple)
    new_w = max(new_w, multiple)
    if (new_h, new_w) == (ori_h, ori_w):
        return image_tensor, (ori_h, ori_w)
    return F.interpolate(image_tensor, (new_h, new_w), mode='bilinear', align_corners=True), (ori_h, ori_w)


def update_confusion(confusion, pred, target, num_classes, ignore_index=255):
    valid = target != ignore_index
    valid &= target >= 0
    valid &= target < num_classes
    valid &= pred >= 0
    valid &= pred < num_classes
    encoded = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_iou(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.full_like(intersection, np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou, np.nanmean(iou)


def evaluate_reference_predictions(model, cfg, input_dir, id_path, reference_masks, resize_multiple, logger):
    model_was_training = model.training
    model.eval()
    input_dir = Path(input_dir)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with open(id_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]

    confusion = np.zeros((cfg['nclass'], cfg['nclass']), dtype=np.int64)
    matched, missing = 0, 0
    with torch.no_grad():
        for idx, sample_id in enumerate(ids, start=1):
            image_rel = image_rel_from_id(sample_id)
            image_path = input_dir / image_rel
            mask_rel = mask_rel_from_image_rel(image_rel)
            reference_path = reference_masks.get(mask_rel)
            if reference_path is None:
                missing += 1
                continue

            image = Image.open(image_path).convert('RGB')
            image_tensor = normalize(image).unsqueeze(0).cuda()
            image_tensor, ori_size = resize_to_multiple(image_tensor, resize_multiple)
            logits = model(image_tensor)
            if logits.shape[-2:] != ori_size:
                logits = F.interpolate(logits, ori_size, mode='bilinear', align_corners=True)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            target = np.array(Image.open(reference_path), dtype=np.uint8)
            if pred.shape != target.shape:
                raise ValueError('Shape mismatch for %s: pred %s, reference %s' % (mask_rel, pred.shape, target.shape))
            update_confusion(confusion, pred, target, cfg['nclass'])
            matched += 1

            if idx % 500 == 0:
                logger.info('Reference mIoU inference: %d/%d images processed' % (idx, len(ids)))

    if model_was_training:
        model.train()
    iou, miou = compute_iou(confusion)
    return miou, iou, matched, missing


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
        reference_root, reference_masks = (None, None)
        if args.reference_pred_dir is not None:
            reference_root, reference_masks = load_reference_masks(args.reference_pred_dir)
            logger.info('Reference prediction masks: %s (%d files)\n' % (reference_root, len(reference_masks)))

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    if not args.no_val:
        valset = SemiDataset(
            cfg['dataset'], cfg['data_root'], 'val', id_path=args.val_id_path
        )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    if not args.no_val:
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(
            valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
        )
    
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    best_reference_miou, best_reference_epoch = 0.0, 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        best_reference_miou = checkpoint.get('best_reference_miou', 0.0)
        best_reference_epoch = checkpoint.get('best_reference_epoch', 0)
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = (loss_x + loss_u_s) / 2.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
        is_best, is_best_ema = False, False
        if not args.no_val:
            eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
            mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
            mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
            
            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                                'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
                
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                    writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

            is_best = mIoU >= previous_best
            is_best_ema = mIoU_ema >= previous_best_ema
            
            previous_best = max(mIoU, previous_best)
            previous_best_ema = max(mIoU_ema, previous_best_ema)
            if mIoU == previous_best:
                best_epoch = epoch
            if mIoU_ema == previous_best_ema:
                best_epoch_ema = epoch
        elif rank == 0:
            logger.info('***** Validation skipped *****\n')
        
        reference_miou, reference_iou = None, None
        is_best_reference = False
        if rank == 0:
            if args.reference_pred_dir is not None:
                eval_model = model_ema.module if hasattr(model_ema, 'module') else model_ema
                reference_id_path = args.reference_id_path or args.unlabeled_id_path
                reference_miou, reference_iou, reference_matched, reference_missing = evaluate_reference_predictions(
                    eval_model, cfg, args.reference_input_dir, reference_id_path,
                    reference_masks, args.reference_resize_multiple, logger
                )
                is_best_reference = reference_miou >= best_reference_miou
                if is_best_reference:
                    best_reference_miou = reference_miou
                    best_reference_epoch = epoch
                logger.info(
                    '***** Reference Prediction Comparison ***** >>>> '
                    'mIoU: %.4f, Best: %.4f @epoch-%d, matched: %d, missing: %d\n'
                    % (reference_miou * 100, best_reference_miou * 100, best_reference_epoch,
                       reference_matched, reference_missing)
                )
                writer.add_scalar('reference/mIoU', reference_miou * 100, epoch)
                for i, iou in enumerate(reference_iou):
                    if not np.isnan(iou):
                        writer.add_scalar('reference/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou * 100, epoch)

            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema,
                'best_reference_miou': best_reference_miou,
                'best_reference_epoch': best_reference_epoch,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                torch.save(checkpoint, os.path.join(args.save_path, 'epoch_%d.pth' % (epoch + 1)))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            if is_best_ema:
                torch.save(checkpoint, os.path.join(args.save_path, 'best_ema.pth'))
            if is_best_reference:
                torch.save(checkpoint, os.path.join(args.save_path, 'best_reference_miou.pth'))

        dist.barrier()


if __name__ == '__main__':
    main()
