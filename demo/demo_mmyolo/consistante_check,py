import os, json, copy, glob, shutil, random, warnings
import numpy as np
import os.path as osp
import argparse
import cv2
import itertools
import mmcv
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import maskUtils
from functools import partial
from multiprocessing import Pool
from terminaltables import AsciiTable
from mmengine.utils import mkdir_or_exist
from mmengine.logging import print_log
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='计算map，并输出预测结果')
    parser.add_argument('gt', help='gt path cocoformat')
    parser.add_argument('pre', help='results path cocoformat')
    parser.add_argument('--save-path', default=None, help='use this path to save analysis results')
    parser.add_argument('--min-scores', default=0.3, type=float)
    parser.add_argument('--area', default=50, type=float, help='custom area evaluate, e.g. [40, 50]')
    parser.add_argument('--include', nargs='+', default=["miss", "error", "right"], type=list)
    parser.add_argument('--img-prefix', default='')
    parser.add_argument('--nproc', default=16, type=int)
    args = parser.parse_args()
    return args

def ncolors(num):
    colors = []
    i = 0
    while i < num:
        color = np.random.choice(range(256), size=3)
        color_out = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        if min(sum((color-color_out)**2).T) < 100: continue
        i += 1
        colors.append([int(i) for i in color])
    return colors

def make_table(headers, results):
        num_columns = len(headers)
        results_flatten = list(
            itertools.chain(*results))
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=None)

def cv2_draw_bbox(img, bboxes, labels, rect_color=(0, 0, 255), fill_color=(0, 0, 0), font_scale=0.5, thickness=2):
    if len(bboxes) == 0: return img
    for bbox, label in zip([bboxes], [labels]):
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        ((text_width, text_height), _) = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

        if min(ymax-ymin, xmax-xmin) < 100: thickness = 1
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), rect_color, thickness)
        if xmax-xmin > 2*text_width:
            cv2.rectangle(img, (xmin, ymin), (xmin + text_width, ymin + int(1.3 * text_height)), fill_color, -1)
            cv2.putText(img, str(label), (xmin, ymin + int(text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), lineType=cv2.LINE_AA)
    return img

def multiprocess_func(func, p_iter, nproc, pool=None, close=True, descr=""):
    if nproc == 1:
        pbar = tqdm(p_iter, total=len(p_iter), ncols=100)
        pbar.set_description(descr)
        results = []
        for _p in pbar:
            p = func(_p)
            if p: results.append(p)
    else:
        pool = Pool(processes=nproc) if pool is None else pool
        pbar = tqdm(pool.imap(func, p_iter), total=len(p_iter), ncols=100)
        pbar.set_description(descr)
        results = []
        for p in pbar:
            if p: results.append(p)
        if close: pool.close()
    return results

def parse_error(error_id, colors, classes, catIds, categories, save_path, img_prefix='', include=["miss", "right", "error"]):
    def loadCats(ids=[]):
        if isinstance(ids, list):
            return [categories[id] for id in ids]
        elif isinstance(ids, int):
            return [categories[ids]]

    filename = osp.join(img_prefix, error_id["file_name"])
    cv2_img = cv2.imread(filename)
    count = {'miss': 0, 'error': 0, 'right': 0}
    color = {'miss': (255, 0, 0), 'right': (0, 255, 0), 'error': (0, 0, 255)}
    if len(error_id["miss"]) + len(error_id["error"]) == 0: return 
    
    for cat_id in catIds:
        nm = loadCats(int(cat_id))[0] if cat_id > -1 else {'name': "agnostic"}
        cls_name = nm['name']
        for i in include:
            for bbox in error_id[i]:
                box = bbox['bbox']
                label_text = loadCats(bbox['category_id'])[
                    0]['name'] if cat_id > -1 else "agnostic"
                fill_color = colors[classes.index(label_text)]
                if cls_name != label_text: continue
                try:
                    score = bbox['score']
                    label_text = "{}-{}".format(label_text, f'{float(score):0.2f}')
                except: pass
                xmin, ymin, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                xmax = xmin + w
                ymax = ymin + h
                cv2_img = cv2_draw_bbox(
                    cv2_img, [xmin, ymin, xmax, ymax], label_text, rect_color=color[i], fill_color=fill_color)
                count[i] += 1

    if count["miss"] > 0:
        mkdir_or_exist(save_path+'/miss')
        cv2.imwrite(osp.join(save_path+'/miss', osp.basename(filename)), cv2_img)
    
    if count["error"] > 0:
        mkdir_or_exist(save_path+'/error')
        cv2.imwrite(osp.join(save_path+'/error', osp.basename(filename)), cv2_img)

def custom_analysis_error(customEval, save_path, img_prefix='', include=["miss", "right", "error"], nproc=16, score_thr=0.5):
    coco_error_dict = {}
    customDt = customEval.customDt
    customGt = customEval.customGt
    for i in customEval.params.imgIds:
        img_info = customEval.loadImgs([i])[0]
        coco_error_dict[img_info['id']] = dict(
            file_name=img_info['file_name'],
            miss=[],
            error=[],
            right=[])

    for eval_ids in customEval.evalImgs:
        if eval_ids is None:
            continue
        aRng = eval_ids['aRng']
        if aRng != customEval.params.areaRng[0]:
            continue
        image_id = eval_ids['image_id']

        dtMatches = eval_ids['dtMatches']
        dtIds = eval_ids['dtIds']
        _, D = dtMatches.shape
        index = np.argwhere(customEval.params.iouThrs == score_thr)[0, 0]
        for d, m in zip(dtIds, dtMatches[index]):
            if m == 0:
                error_dt = customDt[d]
                coco_error_dict[image_id]['error'].append(error_dt)
            else:
                right_gt = customDt[d]
                coco_error_dict[image_id]['right'].append(right_gt)

        gtIds = eval_ids['gtIds']
        gtMatches = eval_ids['gtMatches']
        for g, m in zip(gtIds, gtMatches[index]):
            if m == 0:
                miss_gt = customGt[g]
                coco_error_dict[image_id]['miss'].append(miss_gt)

    classes = customEval.classes
    parse_error_func = partial(
                        parse_error,   
                        colors=ncolors(len(classes)),  
                        classes=classes, 
                        catIds=customEval.params.catIds,
                        categories=customEval.categories,
                        img_prefix=img_prefix, 
                        save_path=save_path, 
                        include=include)
    multiprocess_func(
        parse_error_func, 
        coco_error_dict.values(), 
        nproc=nproc, 
        descr="analysis error")

class Params:
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        self.maxDets = [100]
        self.areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2],
                        [96**2, 1e5**2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self):
        self.setDetParams()

class CustomEval(object):
    def __init__(self, 
        Gt=None, 
        Dt=None, 
        area=[],
        maxDets=100, 
        min_scores=None, 
        nproc=None):

        self.Gt = Gt
        self.Dt = Dt
        self.params = Params()
        self.nproc = nproc
        self.min_scores = min_scores
        self.params.useCats = True

    def loadImgs(self, ids=[]):
        if isinstance(ids, list):
            return [self.imgs[id] for id in ids]
        elif isinstance(ids, int):
            return [self.imgs[ids]]

    def loadCats(self, ids=[]):
        if isinstance(ids, list):
            return [self.categories[id] for id in ids]
        elif isinstance(ids, int):
            return [self.categories[ids]]

    def filter_by_scores(self, Dt):
        new_results = []
        for res in Dt:
            if res['score'] >= self.min_scores:
                new_results.append(res)
        return new_results

    def createIndex_coco(self):

        # process COCO Gt
        Gt = json.load(open(self.Gt, 'r'))
        self.classes = []
        # 基于文件名匹配image_id 
        pathToId = {}
        print('creating index...')
        anns_Gt = {}
        imgToAnns = defaultdict(list)
        self.categories = {}
        self.imgs = {}
        if 'annotations' in Gt:
            for ann in Gt['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns_Gt[ann['id']] = ann
        if 'categories' in Gt:
            for cat in Gt['categories']:
                self.categories[cat['id']] = cat
                self.classes.append(cat['name'])

        if 'images' in Gt:
            for img in Gt['images']:
                self.imgs[img['id']] = img
                pathToId[img['file_name']] = img['id']
        imgIds = list(self.imgs.keys())
        catIds = [cat['id'] for cat in Gt['categories']]
        
        self.params.catIds = catIds
        self.params.imgIds = imgIds

        lists = [imgToAnns[imgId] for imgId in imgIds if imgId in imgToAnns]
        filter_anns = list(itertools.chain.from_iterable(lists))
        filter_anns = filter_anns if len(catIds) == 0 else [
            ann for ann in filter_anns if ann['category_id'] in catIds
        ]
        annIds = [ann['id'] for ann in filter_anns]
        gts = [anns_Gt[id] for id in annIds]

        print(anns_Gt.get(next(iter(anns_Gt))))
        print("Gt"*30)
        self.customGt = anns_Gt

        # process COCO Dt
        Dt = json.load(open(self.Dt, 'r'))
        anns_Dt = {}
        '''
        COCO.json可能来源不同，因此需要基于文件名映射，包括
        id、image_id、category_id
        '''
        if isinstance(Dt, dict):
            
            IdTopath = {}
            for img in Dt['images']:
                IdTopath[img['id']] = img['file_name']
            
            imgToAnns = defaultdict(list)
            for ann in Dt['annotations']:
                ann['category_id'] = self.classes.index(Dt['categories'][ann['category_id']-1]['name']) + 1
                ann['score'] = 1.0
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                # 预标注结果会存在部分图片对应的json文件缺失
                try:
                    ann['image_id'] = pathToId[IdTopath[ann['image_id']]]
                    imgToAnns[ann['image_id']].append(ann)
                    anns_Dt[ann['id']] = ann
                except: pass
            try:
                for img in Dt['images']:
                    img['id'] = pathToId[img['file_name']]
            except: pass

            Dt['categories'] = Gt['categories']

            imgIds = list(self.imgs.keys())

            catIds = [cat['id'] for cat in Dt['categories']]

        elif isinstance(Dt, list):
            Dt = self.filter_by_scores(Dt)
            imgIds_Dt = [ann['image_id'] for ann in Dt]
            assert set(imgIds_Dt) == (set(imgIds_Dt) & set(imgIds)), \
                'Results do not correspond to current coco set'
            for id, ann in enumerate(Dt):
                bb = ann['bbox']
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
            anns_Dt = {}
            imgToAnns = defaultdict(list)
            for ann in Dt:
                imgToAnns[ann['image_id']].append(ann)
                anns_Dt[ann['id']] = ann
            
        else:
            print("数据有误！")
        
        lists = [imgToAnns[imgId] for imgId in imgIds if imgId in imgToAnns] 
        filter_anns = list(itertools.chain.from_iterable(lists))
        filter_anns = filter_anns if len(catIds) == 0 else [
        ann for ann in filter_anns if ann['category_id'] in catIds]  # filter by catIds
        annIds = [ann['id'] for ann in filter_anns]
        dts = [anns_Dt[id] for id in annIds]
        print(anns_Dt.get(next(iter(anns_Dt))))
        print("Dt"*30)
        self.customDt = anns_Dt

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p
        catIds = p.catIds if p.useCats else [-1]
        pool = Pool(processes=self.nproc)
        computeIoU_func = partial(self.computeIoU)
        pool_ious = pool.starmap(
            computeIoU_func,
            [(imgId, catId) for imgId in p.imgIds for catId in catIds])
        self._ious = {
            (imgId, catId): per_iou for imgId, catId, per_iou in pool_ious}
        
        maxDet = p.maxDets[-1]
        evaluateImg_func = partial(self.evaluateImg)
        self.evalImgs = pool.starmap(
            evaluateImg_func,
            [(imgId, catId, areaRng, maxDet)
                for catId in catIds
                for areaRng in p.areaRng
                for imgId in p.imgIds])
        self._paramsEval = copy.deepcopy(self.params)
        pool.close()

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return imgId, catId, []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return imgId, catId, ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self._ious[imgId, catId][:, gtind] if len(
            self._ious[imgId, catId]) > 0 else self._ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }
    
    def accumulate(self):
        p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M)) 
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0: continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:  # noqa: E722
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }

    def _summarize(self, ap=1, catIds=None, iouThr=None, areaRng='all'):
        R = len(self.params.recThrs)
        iStr = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.4f}'  # noqa: E501
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(self.params.iouThrs[0], self.params.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [
            i for i, aRng in enumerate(self.params.areaRngLbl) if aRng == areaRng
        ]
        mind = [i for i, _ in enumerate(self.params.maxDets)]
        if ap == 1:
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == self.params.iouThrs)[0]
                s = s[t]
            if catIds is not None:
                t = []
                catIds_array = np.array(self.params.catIds)
                for catid in catIds:
                    t.extend(np.where(catid == catIds_array)[0].tolist())
                t = np.array(t)
                s = s[:, :, t]
            s = s[:, :, :, aind, mind]
        else:
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == self.params.iouThrs)[0]
                s = s[t]
            if catIds is not None:
                t = []
                catIds_array = np.array(self.params.catIds)
                for catid in catIds:
                    t.extend(np.where(catid == catIds_array)[0].tolist())
                t = np.array(t)
                s = s[:, t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0: mean_s = float("nan")
        else: mean_s = np.mean(s[s > -1])
        return mean_s        

def main():
    args = parse_args()
    customEval = CustomEval(
                Gt=args.gt, 
                Dt=args.pre,
                area=[args.area],
                min_scores=args.min_scores,
                nproc=args.nproc)

    customEval.createIndex_coco()
    customEval.evaluate()
    customEval.accumulate()

    headers = ['category']
    results = ['all_category']
    ap = customEval._summarize(ap=1)
    headers.append("AP")
    results.append(f'{float(ap):0.4f}')

    ap05 = customEval._summarize(ap=1, iouThr=0.5)
    headers.append("AP0.5")
    results.append(f'{float(ap05):0.4f}')

    # for area_str in customEval.params.areaRngLbl[1:]:
    #     ap05_per_area = customEval._summarize(ap=1, iouThr=0.5, areaRng=area_str)
    #     headers.append("AP0.5 {}".format(area_str))
    #     results.append(f'{float(ap05_per_area):0.4f}')

    ar = customEval._summarize(ap=0)
    headers.append("AR")
    results.append(f'{float(ar):0.4f}')

    ar05 = customEval._summarize(ap=0, iouThr=0.5)
    headers.append("AR0.5")
    results.append(f'{float(ar05):0.4f}')

    # for area_str in customEval.params.areaRngLbl[1:]:
    #     ar05_per_area = customEval._summarize(ap=0, iouThr=0.5, areaRng=area_str)
    #     headers.append("AR0.5 {}".format(area_str))
    #     results.append(f'{float(ar05_per_area):0.4f}')

    results = [tuple(results)]

    for idx, catId in enumerate(customEval.params.catIds):
        catId = int(catId)
        nm = customEval.loadCats(catId)[0]
        per_category = [f'{nm["name"]}']
        ap = customEval._summarize(ap=1, catIds=[catId])
        per_category.append(f'{float(ap):0.4f}')

        ap05 = customEval._summarize(ap=1, catIds=[catId], iouThr=0.5)
        per_category.append(f'{float(ap05):0.4f}')

        # for area_str in customEval.params.areaRngLbl[1:]:
        #     ap05_per_area = customEval._summarize(ap=1, catIds=[catId], iouThr=0.5, areaRng=area_str)
        #     per_category.append(f'{float(ap05_per_area):0.4f}')

        ar = customEval._summarize(ap=0, catIds=[catId])
        per_category.append(f'{float(ar):0.4f}')

        ar05 = customEval._summarize(ap=0, catIds=[catId], iouThr=0.5)
        per_category.append(f'{float(ar05):0.4f}')

        # for area_str in customEval.params.areaRngLbl[1:]:
        #     ar05_per_area = customEval._summarize(ap=0, catIds=[catId], iouThr=0.5, areaRng=area_str)
        #     per_category.append(f'{float(ar05_per_area):0.4f}')

        results.append(tuple(per_category))

    make_table(headers, results)

    if args.save_path:
        custom_analysis_error(
                    customEval=customEval, 
                    save_path=args.save_path, 
                    img_prefix=args.img_prefix, 
                    include=args.include, 
                    nproc=args.nproc)

if __name__ == "__main__":
    main()