import cv2
import numpy as np

def DarkChannel(im,sz):
    '''求图像的暗通道：一定区域内所有通道中最小的元素集合
    args:
        im - 图像
        sz - 区域尺寸
    return:
        图像的暗通道图像dark
    '''
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz)) #矩形结构
    dark = cv2.erode(dc,kernel) #腐蚀操作等价于区域内求最小
    return dark

def AtmLight(im,dark):
    '''求大气光（常值）：暗通道中最亮的0.1%像素，对应的接收图像中的最亮的像素
    args:
        im - 雾霾图像
        dark - 暗通道图像
    return:
        大气光常值A
    '''
    num = np.int_(dark.size*0.001) #前0.1%的数量
    darkvec = dark.reshape(-1) #将dark暗通道变成一个向量
    imvec = im.reshape(-1,3) #将BGR三个通道变成三个向量

    index = np.argsort(-darkvec)[:num] #排序后取最大的num个值，获得对应的索引
    index_big = np.argmax(np.sum(imvec,1)[index]) #最亮的像素意为RGB加一起最大，获得最大值的索引
    A = imvec[index_big,:]
    A = np.clip(A, 0, 0.85) #对A做约束，防止图像不自然
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95 #微调
    im3 = np.empty(im.shape,im.dtype)
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[ind]
    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,t_hat):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, t_hat, r, eps)
    return t

def Recover(im, A, t, tx = 0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx) #保留一定的haze
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind] - A[ind])/t + A[ind]
    return res

if __name__ == '__main__':
    path = './image/foggy2.jpg'
    img_raw = cv2.imread(path)
    img_haze = img_raw.astype('float64')/255 #归一化为0-1

    dark = DarkChannel(img_haze, 15)
    A = AtmLight(img_haze, dark)
    t_hat = TransmissionEstimate(img_haze, A, 15)
    t = TransmissionRefine(img_raw, t_hat)
    J = Recover(img_haze, A, t, 0.1)
    J = abs(J) #纠正极小的负值
    J_show = (J/np.max(J)) ** (2/3) #提高曝光
    J_show = np.uint8(J_show*255)
    cv2.imwrite("./image/J_show2.png",J_show)

