import cv2
import numpy as np


def postproccesing(result, thresh):
    new_result = []
    for (coord, text, prob) in result:
        if prob < thresh: continue

        text = text.strip().lower().replace('.', ' ')

        # sim = is_similarity(text,  mark_words, similarity_thresh)
        # if sim: continue
        if "+" in text: continue
        for txt in text.split(" "):
            txt = txt.strip()
            # print(txt, len(txt), txt.isdigit())
            filtered_txt = "".join(list(filter(lambda x: x.isdigit(), txt)))
            if filtered_txt.isdigit() and  abs(10 - len(filtered_txt)) <= 0 and abs(len(filtered_txt) - len(txt)) <= 2:
                new_result.append([coord, filtered_txt, prob])

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