import numpy as np


def box_iou_calc(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min = 0, a_max = None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def read_txt(txt_file, pred=True):
    '''
    Parameters
    ----------
    txt_file : txt file path to read
    pred : if your are raedinf prediction txt file than it'll have 5 values 
    (i.e. including confdience) whereas GT won't have confd value. So set it
    to False for GT file. The default is True.
    Returns
    -------
    info : a list haing 
        if pred=True => detected_class, confd, x_min, y_min, x_max, y_max
        if pred=False => detected_class, x_min, y_min, x_max, y_max
    '''
    x = []
    with open(txt_file, 'r') as f:
        info = []
        x = x + f.readlines()
        for item in x:
            item = item.replace("\n", "").split(" ")
            if pred == True:
                # for preds because 2nd value in preds is confidence
                det_class = item[0]
                confd = float(item[1])
                x_min = int(item[2])
                y_min = int(item[3])
                x_max = int(item[4])
                y_max = int(item[5])
                
                info.append((x_min, y_min, x_max, y_max, confd, det_class))
                            
            else:
                # for preds because 2nd value in preds is confidence
                det_class = item[0]
                x_min = int(float(item[1]))
                y_min = int(float(item[2]))
                x_max = int(float(item[3]))
                y_max = int(float(item[4]))
                
                info.append((det_class, x_min, y_min, x_max, y_max))
                
        return info
    
def IoU(target_boxes , pred_boxes):
    xA = np.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = np.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = np.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = np.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = np.maximum(0.0, xB - xA ) * np.maximum(0.0, yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    iou = np.nan_to_num(iou)
    return iou    

def process(x, class_names, gt = True):
    '''
    Parameters
    ----------
    x : class_name, x_min, y_min, x_max, y_max
    Returns
    -------
    x : class_index, x_min, y_min, x_max, y_max
    '''
    if gt:
        clas = x[:,0]
        temp = []
        for i in range(len(clas)):
            temp.append(class_names.index(clas[i]))
        temp = np.array(temp)
        x[:,0] = temp
        x = x.astype(np.int32)
    else:
        clas = x[:,-1]
        temp = []
        for i in range(len(clas)):
            try:
                temp.append(class_names.index(clas[i]))
            except:
                temp.append(class_names.index("NOT"))
        temp = np.array(temp)
        x[:,-1] = temp
        x = x.astype(np.float32)
    
    return x
