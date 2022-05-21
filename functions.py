import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread, imsave
import PIL.Image

def my_read(name): #считывание картинки и приведение к RGB, если надо
    new_img = imread(name)
    w,h,c = new_img.shape
    if c==4:
        rgba_image = PIL.Image.open(name)
        rgb_img = rgba_image.convert('RGB')
        pix = np.array(rgb_img.getdata()).reshape(rgb_img.size[0], rgb_img.size[1], 3)
        return pix
    elif c==3:
        return new_img
    else:
        raise TypeError


def my_filter(img, kmeans, to_name): #фильтр картинки о существующей модели и сохранение результата
    identified_palette = np.array(kmeans.cluster_centers_).astype(int)

    w, h, _ = img.shape
    new_img = img.reshape(w * h, 3)

    labels = kmeans.predict(new_img)

    recolored_img = np.copy(new_img)
    for index in range(len(recolored_img)):
        recolored_img[index] = identified_palette[labels[index]]

    # reshape for display
    recolored_img = recolored_img.reshape(w, h, 3)

    imsave(to_name, recolored_img)


def my_model(n_colors, name): #построение модели
    img = my_read(name)
    w, h, _ = img.shape
    img = img.reshape(w * h, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img)
    # the cluster centroids is our color palette
    # identified_palette = np.array(kmeans.cluster_centers_).astype(int)
    return kmeans