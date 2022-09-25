import PIL
import pathlib
import cv2

import numpy as np
import tensorflow            as tf
import xml.etree.ElementTree as ET


class MyDataset(tf.data.Dataset):

    def __new__(cls, path):
        return cls.from_generator(
            generator    = lambda: cls.generator(path),
            output_types = (tf.float64 , tf.float64),
            output_shapes= ([320,320,3], [4]       ),
            # output_shapes= ([320,320,3], [2,2]     ),
        )


    @classmethod
    def generator(cls, path):

        for p in pathlib.Path(path).iterdir():
            if p.suffix != '.xml'         : continue
            if 1 < len(ET.parse(str(p)).getroot().findall('object')): continue
            img, box = cls.parseAnnotation(p)
            
            yield cls.getImage(img), box

    @classmethod
    def getImage(cls, path):

        img = PIL.Image.open(path)
        img = img.resize([320,320])
        img = np.array(img)
        img = img[:,:,:3]
        img = img / 127.5 - 1

        return img

    @classmethod
    def parseAnnotation(cls, path):
        ano = ET.parse(str(path)).getroot()

        img = ano.find('filename').text
        img = f'{path.parent}/{img}'

        siz = ano.find('size')
        siz = [siz.find('height'), siz.find('width')]
        siz = [e.text for e in siz]
        siz = np.array(siz)
        siz = siz.astype(int)

        bnd = ano.find('object').find('bndbox')
        bnd = [
            bnd.find('ymin'), bnd.find('xmin'),
            bnd.find('ymax'), bnd.find('xmax'),
        ]
        bnd = [e.text for e in bnd]
        bnd = np.array(bnd)
        bnd = bnd.astype(float)
        bnd = bnd.reshape(2,2)
        bnd /= siz
        bnd = bnd.reshape(4)

        return img, bnd