import glob
import os

import numpy as np

from confusion_matrics import read_txt, box_iou_calc


def levenshtein(str_1, str_2):
    n, m = len(str_1), len(str_2)
    if n > m:
        str_1, str_2 = str_2, str_1
        n, m = m, n

    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if str_1[j - 1] != str_2[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]

def num_correctly_rec_char(pred, target):
    i = 0
    acc = 0
    while i < min(len(pred), len(target)):
        acc += int(pred[i] == target[i])
        i += 1

    return acc


gt = glob.glob(os.path.join('/mnt/c/Users/User/Desktop/study/OCR_it_fabric/ocr_dataset/anns','*.txt'))
pred = glob.glob(os.path.join('/mnt/c/Users/User/Desktop/study/OCR_it_fabric/ocr_dataset/preds_doctr','*.txt'))

total_av_lev = 0
num_char = 0
tp = 0
fp = 0
fn = 0
cnt = 0

for i in range(len(gt)):

    y_t = np.asarray(read_txt(gt[i], pred=False))
    y_p = np.asarray(read_txt(pred[i], pred=True))

    for t in y_t:
        flg = False
        num_char += len(t[0])
        for p in y_p:
            if box_iou_calc(p[None, :4].astype(np.int32), t[None, 1:].astype(np.int32)).item() > 0.2:
                # print(p[5], t[0], levenshtein(p[5], t[0]))
                total_av_lev += levenshtein(p[5], t[0])
                cnt += 1
                tp += num_correctly_rec_char(p[5], t[0])
                fp += (len(p[5]) - num_correctly_rec_char(p[5], t[0]))
                fn += (len(t[0]) - num_correctly_rec_char(p[5], t[0]))
                flg = True
        if not flg:
            fn += len(t[0])


# total_av_lev /= len(gt)
print(tp, fp, fn, num_char)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("total av lev", total_av_lev / cnt)
print("CRR", (num_char - total_av_lev) / num_char)
print("Accuracy", tp / num_char)
print("Precision", precision)
print("Recall", recall)
print("F1", 2 * precision * recall / (precision + recall))