import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from skimage import measure


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]


    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = metrics.auc(fprs, pros[idxes])
    return pro_auc


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    
    rec_preds = outputs["rec_pred"].cpu().numpy()  # B x 1 x H x W 
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    if outputs["ref_pred"].numel():
        ref_preds = outputs["ref_pred"].cpu().numpy()
    else:
        ref_preds = np.array([])

    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        if ref_preds.size:
            np.savez(
                save_file,
                filename=filenames[i],
                rec_pred=rec_preds[i],
                mask=masks[i],
                ref_pred= ref_preds[i],
                height=heights[i],
                width=widths[i],
                clsname=clsnames[i],
            )
        else:
            np.savez(
                save_file,
                filename=filenames[i],
                rec_pred=rec_preds[i],
                mask=masks[i],
                height=heights[i],
                width=widths[i],
                clsname=clsnames[i],
            )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    rec_preds = []
    ref_preds = []
    #dis_preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        rec_preds.append(npz["rec_pred"])

        masks.append(npz["mask"])
        if "ref_pred" in npz:
            ref_preds.append(npz["ref_pred"])

    rec_preds = np.concatenate(np.asarray(rec_preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    if ref_preds:
        ref_preds = np.concatenate(np.asarray(ref_preds), axis=0)  # N x H x W 
    else:
        ref_preds = np.array(ref_preds)
    return fileinfos, rec_preds, ref_preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int32)  # change np.int to np.int32 for adopting >numpy 1.20
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        ap = metrics.average_precision_score(self.masks, self.preds, pos_label=1, average=None)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        
        '''
        precisions, recalls, thresholds = metrics.precision_recall_curve(self.masks, self.preds)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_max = np.max(f1_scores[np.isfinite(f1_scores)])
        '''
        
        return auc, ap


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        
        self.preds = data_meta.preds
        self.masks = data_meta.masks
        self.masks[self.masks > 0] = 1
         
        '''
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
        '''
    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks.ravel(), self.preds.ravel(), pos_label=1)
        ap = metrics.average_precision_score(self.masks.ravel(), self.preds.ravel(), pos_label=1, average=None)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        '''
        precisions, recalls, thresholds = metrics.precision_recall_curve(self.masks.ravel(), self.preds.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_max = np.max(f1_scores[np.isfinite(f1_scores)])
        
        pro = cal_pro_score(self.masks, self.preds)
        '''
        return auc, ap 


eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
}


def performances(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.get("auc", None):
            for metric in config.auc:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc, ap = eval_method.eval_auc()
                '''
                if evalname == "max":
                   auc, ap, f1_max = eval_method.eval_auc()
                else:
                   auc, ap, f1_max, pro = eval_method.eval_auc()
                '''

                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc*100
                ret_metrics["{}_{}_ap".format(clsname, evalname)] = ap*100
                '''
                ret_metrics["{}_{}_f1max".format(clsname, evalname)] = f1_max
                if evalname == "pixel":
                   ret_metrics["{}_{}_pro".format(clsname, evalname)] = pro
                '''
                #auc, ap = eval_method.eval_auc()
                #ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc
                #ret_metrics["{}_{}_ap".format(clsname, evalname)] = ap

    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc
            
            evalvalues_ap = [
                ret_metrics["{}_{}_ap".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_ap = np.mean(np.array(evalvalues_ap))
            ret_metrics["{}_{}_ap".format("mean", evalname)] = mean_ap


            '''
            evalvalues_f1max = [
                ret_metrics["{}_{}_f1max".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_f1max = np.mean(np.array(evalvalues_f1max))
            ret_metrics["{}_{}_f1max".format("mean", evalname)] = mean_f1max

            if evalname == "pixel":
               evalvalues_pro = [
                   ret_metrics["{}_{}_pro".format(clsname, evalname)]
                   for clsname in clsnames
               ]
               mean_pro = np.mean(np.array(evalvalues_pro))
               ret_metrics["{}_{}_pro".format("mean", evalname)] = mean_pro
            '''

    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = sorted(list(clsnames - set(["mean"]))) + ["mean"]
    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "_ap" in k or "_auc" in k]
        evalnames = ['max_auc', 'max_ap', 'pixel_auc', 'pixel_ap']

        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
