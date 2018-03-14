import nibabel
import matplotlib.pyplot as plt
import os
import numpy as np

data_dir='path_to_image_folder'                                                 #image directory
img=nibabel.load(os.path.join(data_dir,'image.name'))                           #loading the image
img_data=img.get_data()

hist,bins=np.histogram(img_data[img_data!=0],bins='auto',density=True)         #mapping the histogram of the image using probability density function(density=True), background black values are ignored.
bins=0.5*(bins[1:]+bins[:-1])                                                  #taking midpoints of bins

t1=0                                                                           #threshold1 index
t2=0                                                                           #threshold2 index

currvar=0                                                                      #we have to maximise this value
u=np.zeros(3)                                                                  #mean of the three distributions
w=np.zeros(3)                                                                  #weightages of the three distributions

uT=np.vdot(bins,hist)/np.sum(hist)                                             #mean of the full histogram

for i in range(1,int(len(hist)/2)):
    w[0]=np.sum(hist[:i])/np.sum(hist)
    u[0]=np.vdot(bins[:i],hist[:i])/np.sum(hist[:i])
    for j in range(i+1,len(hist)):
        w[1]=np.sum(hist[i:j])/np.sum(hist)
        u[1]=np.vdot(bins[i:j],hist[i:j])/np.sum(hist[i:j])
        w[2] = np.sum(hist[j:])/np.sum(hist)
        u[2] =np.vdot(bins[j:],hist[j:])/np.sum(hist[j:])
        maxvar=np.vdot(w,(np.power((u-uT),2)))                                  #according to formula
        if(maxvar>currvar):                                                     #maximimsing currvar
            currvar=maxvar
            print(currvar)
            t2 = i
            t1 = j

plt.bar(bins,hist,width=1)
plt.axvline(bins[t1],c='r')                                                     #plotting histogram with the two thresholds,red vertical line is threshold1 and green vertical line is threshold2
plt.axvline(bins[t2],c='g')
plt.show()

threshold1=bins[t1]
threshold2=bins[t2]
print('threshold1 = '+threshold1)
print('threshold2 = '+threshold2)

