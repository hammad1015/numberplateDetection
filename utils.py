import numpy as np

def drawBoxes(imgs, boxs):
    i = np.arange(imgs.shape[0])

    boxs = boxs.reshape(-1,2,2)
    boxs = boxs * imgs.shape[1:3] - 1
    boxs = boxs.astype(int)
    boxs = boxs.reshape(-1,4)

    imgs = imgs - imgs.min()
    imgs = imgs / imgs.max()

    imgs[i, boxs[:,0],:] = [1,0,0]
    imgs[i, boxs[:,2],:] = [1,0,0]
    imgs[i, :,boxs[:,1]] = 1
    imgs[i, :,boxs[:,3]] = 1

    return imgs

def extractNumberplate(img, box):
    img = img[0]
    box = box[0]

    box = box.reshape(2,2)
    box = box * img.shape[:2]
    box = box.reshape(-1)
    box = box.astype(int)

    img = img - img.min()
    img = img / img.max()
    img = img[box[0]:box[2], box[1]:box[3]]

    return img