import cv2
import numpy as np


def try_merge_boxes(boxes, thresh=4):
    new_boxes = []
    # boxes = list(filter(lambda x: str.isdigit(x[1].strip()), boxes))
    for i in range(len(boxes) - 1):
        merged = False
        box1 = boxes[i][0]
        x1min = box1[:, 0].min()
        x1max = box1[:, 0].max()
        y1min = box1[:, 1].min()
        y1max = box1[:, 1].max()
        for j in range(i+1, len(boxes)):
            box2 = boxes[j][0]
            x2min = box2[:, 0].min()
            x2max = box2[:, 0].max()
            y2min = box1[:, 1].min()
            y2max = box1[:, 1].max()
            if (abs(x1max - x2min) <= thresh or abs(x2max - x1min) <= thresh) \
                and abs(y1min - y2min) <= thresh and abs(y1max - y2max) <= thresh:
                x1 = min(x2min, x1min)
                y1 = min(y1min, y2min)

                x3 = max(x2max, x1max)
                y3 = max(y1max, y2max)

                x4 = x1
                y4 = y3

                x2 = x3
                y2 = y1

                if x2max > x1max:
                    new_text = boxes[i][1].strip() +  boxes[j][1].strip()
                else:
                    new_text = boxes[j][1].strip() +  boxes[i][1].strip()

                new_boxes.append((
                    np.array([
                        [x1, y1],
                        [x2, y2],
                        [x3, y3],
                        [x4, y4]
                    ], dtype=np.int64),
                    new_text,
                    (boxes[i][2] + boxes[j][2]) / 2
                ))

                x1min = x1
                x1max = x3
                merged = True
        if not merged:
            new_boxes.append(boxes[i])

    return new_boxes


def postproccesing(result, thresh):
    new_result = []
    for (coord, text, prob) in result:
        if prob < thresh: continue

        text = text.strip().lower().replace('.', ' ')

        # sim = is_similarity(text,  mark_words, similarity_thresh)
        # if sim: continue
        if "+" in text:
            # or ":" in text:
            continue
        len_text = len(text)
        cur_pos = -1
        for txt in text.split(" "):
            cur_pos += 1
            cur_step = len(txt)
            txt = txt.strip()
            # print(txt, len(txt), txt.isdigit())
            filtered_txt = "".join(list(filter(lambda x: not x.isalpha(), txt)))
            if "47" == filtered_txt[:2]: break # обычно это номер телефона, "+" определяется как "4"
            if filtered_txt.isdigit() and  abs(10 - len(filtered_txt)) <= 1 and abs(len(filtered_txt) - len(txt)) <= 2:
                w = coord[1, 0] - coord[0, 0]
                coord[0, 0] += int(w * (cur_pos / len_text))
                coord[-1, 0] += int(w * (cur_pos / len_text))
                new_result.append([coord, filtered_txt, prob])
            cur_pos += cur_step


    if len(new_result) == 0:
        new_result = try_merge_boxes(result)
        if len(new_result) > 0:
            new_result = postproccesing(try_merge_boxes(result), thresh)

    return new_result

def save_detecting(result, img, img_path):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    # img = cv2.imread(img_path)
    predictions = ""
    for id, (coord, text, prob) in enumerate(result):
        (topleft, topright, bottomright, bottomleft) = coord
        tx,ty = (int(topleft[0]), int(topleft[1]))
        bx,by = (int(bottomright[0]), int(bottomright[1]))
        cv2.rectangle(img, (tx,ty), (bx,by), (0, 0, 255), 2)
        # print(str(id + 1) + ")  " + str(text))
        cv2.putText(img, str(id + 1) + ")  " + str(text), (topleft[0], topleft[1]-10), font,
        fontScale, color, thickness, cv2.LINE_AA)
        predictions += " ".join(map(str, [text,prob,tx,ty,bx,by])) + '\n'
    cv2.imwrite(img_path+ ".jpg", img)
    with open(img_path+ ".txt", 'w') as f:
        f.write(predictions)


def save_results(result, path: str) -> None:
    target = []
    for id, (coord, text, prob) in enumerate(result[:1]):
        (topleft, topright, bottomright, bottomleft) = coord
        tx,ty = (int(topleft[0]), int(topleft[1]))
        bx,by = (int(bottomright[0]), int(bottomright[1]))
        target.append([tx,ty,bx,by,text])
    np.save(path, target)