import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img_m
import seaborn as sns
from scipy.cluster.vq import kmeans, vq
import sys
from PIL import Image

#########################
#the following code turns any image on a white background into a pixelated image
#simply type pixelartilize(fname) with fname being any image file
#
#
#Current limitations: content must be on a pure white background for canny egde to work
#
#Currently not understood: K values are non deterministic
#
#########################


###Following mosiac image code section taken from: https://github.com/chschommer/pixelate-images/blob/main/mosaic.py

def avg_pixel_color_for_block(image, i_start, j_start, i_end, j_end):
    count = 0
    (r,g,b) = (0,0,0)

    if (i_end > image.width):
        i_end = image.width

    if (j_end > image.height):
        j_end = image.height
   
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            (r_temp, g_temp, b_temp) = image.getpixel((i, j))
            if r_temp == 255 or g_temp==255 or b_temp==255:
                continue
            (r,g,b) = (r+r_temp, g+g_temp, b+b_temp)
            count = count+1
    try:
        return (int(r/count), int(g/count), int(b/count))
    except:
        return(255,255,255)


def fill_block(image, color, i_start, j_start, i_end, j_end):
    if (i_end > image.width):
        i_end = image.width

    if (j_end > image.height):
        j_end = image.height

    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            image.putpixel((i,j), color)


def do_moasic(image, block_size):
    result = Image.new('RGB', (image.width, image.height), color = 'black')

    for i in range(0, image.width, block_size):
        for j in range(0, image.height, block_size):
            avg_color = avg_pixel_color_for_block(image, i, j, i + block_size, j + block_size)
            fill_block(result, avg_color, i,j, i+block_size, j+block_size)


    return result 


##function retrieved from https://python-bloggers.com/2020/08/get-the-dominant-colors-of-an-image-with-k-means/
def k_stuff(img):
    r_arr=[]
    g_arr=[]
    b_arr=[]
    h=np.shape(img)[0]
    w=np.shape(img)[1]
    for y in range(h):
        for x in range(w):
            # A pixel contains RGB values
            (r, g, b) = img.getpixel((x,y))
            r_arr.append(r)
            g_arr.append(g)
            b_arr.append(b)
    
    df = pd.DataFrame({'red':r_arr, 'green':g_arr, 'blue':b_arr})
    df.head()
    distortions = []
    num_clusters = range(1, 7)
    # Create a list of distortions from the kmeans function
    for i in num_clusters:
        cluster_centers, distortion = kmeans(df[['red','green','blue']].values.astype(float), i)
        distortions.append(distortion)
        
    elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

    cluster_centers, _ = kmeans(df[['red','green','blue']].values.astype(float), 5)
    return cluster_centers




def get_distance(pi,p2):
    return ((pi[0]-p2[0])**2+(pi[1]-p2[1])**2+(pi[1]-p2[1])**2)**.5 #Get distance

def round_img(img,palette):
    h=np.shape(img)[0]
    w=np.shape(img)[1]
    for y in range(h):
        for x in range(w):
            # A pixel contains RGB values
            mini = None
            (r, g, b) = img.getpixel((x,y))
            for p in palette:
                dist = get_distance((r,g,b),p)#get the closest rgb value in 3d space
                if mini==None or dist < mini:
                    mini=dist
                    pal=p

            #set colors to closest color
            r=int(pal[0])
            g=int(pal[1])
            b=int(pal[2])

            img.putpixel((x,y), (r,g,b))#set pixel to that color
    return img

def crop_image(img,edges):
    h=np.shape(edges)[0]
    w=np.shape(edges)[1]
    
    top=h
    bottom=0
    left=w
    right=0
    for y in range(h):
        for x in range(w):
            #find edges of content
            value=edges[y][x]
            if value==255 and x < left:
                left = x
            if value==255 and x > right:
                right = x
            if value==255 and y < top:
                top = y
            if value==255 and y > bottom:
                bottom = y


    crop = img[top:bottom, left:right]
    return crop


def pixelartilize(fname):
    image = cv2.imread(fname)
 
    greyscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
     
    canny_edge = cv2.Canny(greyscale, threshold1=30, threshold2=100)

    crop = crop_image(image,canny_edge)

    new_width = np.shape(crop)[0]
    new_height =  np.shape(crop)[0]
    block_size = max((new_width),(new_height))//32

    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)

    target = do_moasic(im, block_size)


    with Image.open(fname) as im:
        pallette=k_stuff(im)

    final = round_img(target,pallette)
    final.save("pixelated_"+fname)

if __name__ == "__main__":
    fname = input("Please input the filename you wish to pixelize: ")
    try:
        pixelartilize(fname)
        input("saved to pixelated_"+fname+", press enter to quit")
    except:
        input("Unkown error encountered, enter to quit")
    

