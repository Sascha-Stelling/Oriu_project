
# coding: utf-8

# In[370]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[533]:



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[456]:


import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[534]:


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            #x3=x
            x = self.pool(F.relu(self.conv2(x)))
            #x2=x
            x = self.pool(F.relu(self.conv3(x)))
            x1=x
            x = x.view(-1, 32 * 2 * 2)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x,x1


net = Net()


# In[535]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[536]:


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, outputs1 = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# In[526]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x3=x
        x = self.pool(F.relu(self.conv2(x)))
        #x2=x
        x = self.pool(F.relu(self.conv3(x)))
        x1=x
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x,x1


net = Net()


# In[506]:


np.shape(outputs1)


# In[543]:


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x2=x
        x = self.pool(F.relu(self.conv2(x)))
        x1=x
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x,x1


# In[545]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    layer1=[]
    l=[]
    layer2=[]
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        l.append(labels[0])
        l.append(labels[1])
        l.append(labels[2])
        l.append(labels[3])
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, outputs1 = net(inputs)
        
        layer1.append(outputs2[0,1,:,:].detach().numpy())
        layer1.append(outputs2[1,1,:,:].detach().numpy())
        layer1.append(outputs2[2,1,:,:].detach().numpy())
        layer1.append(outputs2[3,1,:,:].detach().numpy())
        
        #layer2.append(outputs1[0,1,:,:].detach().numpy())
        #layer2.append(outputs1[1,1,:,:].detach().numpy())
        #layer2.append(outputs1[2,1,:,:].detach().numpy())
        #layer2.append(outputs1[3,1,:,:].detach().numpy())
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# In[442]:


#np.shape(layer2)
#train_SVM=np.reshape(layer2,(500
np.shape(outputs)


# In[ ]:


from sklearn import svm
#from sklearn.metrics import hinge_loss
clf = svm.SVC(kernel='linear', C=1)

        
clf.fit(train_SVM,l)


# In[70]:


C=outputs1[3,8,:,:].detach().numpy()
D=np.reshape(C,(1,25))
clf.predict(D)
classes[8]
l[3]


# In[35]:


np.shape(A)


# In[38]:


A=outputs1[1,8,:,:]


# In[47]:


np.reshape(A,(1000,25))


# In[84]:


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=2)
dataiter = iter(testloader)
images, labels = dataiter.next()


# In[85]:


np.shape(images)


# In[399]:


test_svm=[]
l=[]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs,outputs1,outputs2 = net(images)
        test_svm.append(outputs1[0,1,:,:].detach().numpy())
        l.append(labels)
        


# In[408]:


test_svm=np.reshape(test_svm,(10000,25))
np.shape(test_svm)


# In[386]:


test_svm=outputs1.detach().numpy()


# In[385]:


labels=labels.detach().numpy()


# In[409]:


clf.score(test_svm, l)


# In[121]:


image1=images[1].detach().numpy()
labels[1]
print (classes[8])


# In[115]:


image1 = np.transpose(image1, (1, 2, 0))


# In[116]:


np.shape(image1)


# In[117]:


plt.imshow(image1)


# In[126]:


outputs1[1,1,:,:]


# In[232]:


image2=outputs2[1,1,:,:].detach().numpy()


# In[233]:


from skimage import color
image2=color.rgb2gray(image2)


# In[234]:


plt.imshow(image2)


# In[184]:


image3=outputs1[1,1,:,:].detach().numpy()
from skimage import color
image2=color.rgb2gray(image3)
plt.imshow(image3)


# In[165]:


classes[3]


# In[236]:


image7=images[500].detach().numpy()
labels[500]
print (classes[labels[500]])
image7 = np.transpose(image7, (1, 2, 0))
plt.imshow(image7)


# In[237]:


image8=outputs2[500,1,:,:].detach().numpy()
plt.imshow(image8)image8=outputs2[500,1,:,:].detach().numpy()
plt.imshow(image8)


# In[211]:


np.shape(image1)


# In[327]:


image1=images[1].detach().numpy()
labels[1]
#print (classes[8])
image1 = np.transpose(image1, (1, 2, 0))
image1_gray=color.rgb2gray(image1)
np.shape(image1_gray)


# In[345]:


normalize = False  
orientations=9
block_norm='L2-Hys'
pixels_per_cell = [8, 8]  
cells_per_block = [2, 2]
def extractFeature(img, vis=False):
    from skimage.feature import hog
    return hog(img, orientations, pixels_per_cell, cells_per_block, block_norm, visualise=vis, transform_sqrt=normalize)


# In[344]:


f, arr = plt.subplots(3,10, figsize=(16,5))
arr[0,0].imshow(image1)
arr[0,0].axis('off')
arr[1,0].imshow(image2)
arr[1,0].axis('off')
arr[2,0].imshow(image3)
arr[2,0].axis('off')
arr[0,1].imshow(image4)
arr[0,1].axis('off')
arr[1,1].imshow(image5)
arr[1,1].axis('off')
arr[2,1].imshow(image6)
arr[2,1].axis('off')
arr[0,2].imshow(image7)
arr[0,2].axis('off')
arr[1,2].imshow(image8)
arr[1,2].axis('off')
arr[2,2].imshow(image9)
arr[2,2].axis('off')
arr[0,3].imshow(image10)
arr[0,3].axis('off')
arr[1,3].imshow(image11)
arr[1,3].axis('off')
arr[2,3].imshow(image12)
arr[2,3].axis('off')
arr[0,4].imshow(image13)
arr[0,4].axis('off')
arr[1,4].imshow(image14)
arr[1,4].axis('off')
arr[2,4].imshow(image15)
arr[2,4].axis('off')
arr[0,5].imshow(image16)
arr[0,5].axis('off')
arr[1,5].imshow(image17)
arr[1,5].axis('off')
arr[2,5].imshow(image18)
arr[2,5].axis('off')
arr[0,6].imshow(image19)
arr[0,6].axis('off')
arr[1,6].imshow(image20)
arr[1,6].axis('off')
arr[2,6].imshow(image21)
arr[2,6].axis('off')
arr[0,7].imshow(image22)
arr[0,7].axis('off')
arr[1,7].imshow(image23)
arr[1,7].axis('off')
arr[2,7].imshow(image24)
arr[2,7].axis('off')
arr[0,8].imshow(image25)
arr[0,8].axis('off')
arr[1,8].imshow(image26)
arr[1,8].axis('off')
arr[2,8].imshow(image27)
arr[2,8].axis('off')
arr[0,9].imshow(image28)
arr[0,9].axis('off')
arr[1,9].imshow(image29)
arr[1,9].axis('off')
arr[2,9].imshow(image30)
arr[2,9].axis('off')


# In[228]:


image11=images[7891].detach().numpy()
labels[7891]
print (classes[labels[7891]])
image11 = np.transpose(image11, (1, 2, 0))


# In[240]:


image9=outputs1[500,1,:,:].detach().numpy()
plt.imshow(image9)


# In[245]:


image10=images[3971].detach().numpy()
labels[3971]
print (classes[labels[3971]])
image10 = np.transpose(image10, (1, 2, 0))
plt.imshow(image10)


# In[246]:


image11=outputs2[3971,1,:,:].detach().numpy()
plt.imshow(image11)


# In[247]:


image12=outputs1[3971,1,:,:].detach().numpy()
plt.imshow(image12)


# In[306]:


f1, arr1 = plt.subplots(1,3, figsize=(16,5))
image13=images[391].detach().numpy()
labels[391]
print (classes[labels[391]])
image13 = np.transpose(image13, (1, 2, 0))
#plt.imshow(image10)
image14=outputs2[391,1,:,:].detach().numpy()
#plt.imshow(image11)
image15=outputs1[391,1,:,:].detach().numpy()
#plt.imshow(image12)

arr1[0].imshow(image13)
arr1[0].axis('off')
arr1[1].imshow(image14)
arr1[1].axis('off')
arr1[2].imshow(image15)
arr1[2].axis('off')


# In[260]:


f2, arr2 = plt.subplots(1,3, figsize=(16,5))
image16=images[190].detach().numpy()
labels[190]
print (classes[labels[190]])
image16 = np.transpose(image16, (1, 2, 0))
#plt.imshow(image10)
image17=outputs2[190,1,:,:].detach().numpy()
#plt.imshow(image11)
image18=outputs1[190,1,:,:].detach().numpy()
#plt.imshow(image12)

arr2[0].imshow(image16)
arr2[0].axis('off')
arr2[1].imshow(image17)
arr2[1].axis('off')
arr2[2].imshow(image18)
arr2[2].axis('off')


# In[254]:


f3, arr3 = plt.subplots(1,3, figsize=(16,5))
image19=images[1987].detach().numpy()
labels[1987]
print (classes[labels[1987]])
image19 = np.transpose(image19, (1, 2, 0))
#plt.imshow(image10)
image20=outputs2[1987,1,:,:].detach().numpy()
#plt.imshow(image11)
image21=outputs1[1987,1,:,:].detach().numpy()
#plt.imshow(image12)

arr3[0].imshow(image19)
arr3[0].axis('off')
arr3[1].imshow(image20)
arr3[1].axis('off')
arr3[2].imshow(image21)
arr3[2].axis('off')


# In[259]:


f5, arr5 = plt.subplots(1,3, figsize=(16,5))
image4=images[19].detach().numpy()
labels[19]
print (classes[labels[19]])
image4 = np.transpose(image4, (1, 2, 0))
#plt.imshow(image10)
image5=outputs2[19,1,:,:].detach().numpy()
#plt.imshow(image11)
image6=outputs1[19,1,:,:].detach().numpy()
#plt.imshow(image12)

arr5[0].imshow(image4)
arr5[0].axis('off')
arr5[1].imshow(image5)
arr5[1].axis('off')
arr5[2].imshow(image6)
arr5[2].axis('off')


# In[281]:


f6, arr6 = plt.subplots(1,3, figsize=(16,5))
image22=images[114].detach().numpy()
labels[114]
print (classes[labels[114]])
image22 = np.transpose(image22, (1, 2, 0))
#plt.imshow(image10)
image23=outputs2[114,1,:,:].detach().numpy()
#plt.imshow(image11)
image24=outputs1[114,1,:,:].detach().numpy()
#plt.imshow(image12)

arr6[0].imshow(image22)
arr6[0].axis('off')
arr6[1].imshow(image23)
arr6[1].axis('off')
arr6[2].imshow(image24)
arr6[2].axis('off')


# In[271]:


f7, arr7 = plt.subplots(1,3, figsize=(16,5))
image25=images[808].detach().numpy()
labels[808]
print (classes[labels[808]])
image25 = np.transpose(image25, (1, 2, 0))
#plt.imshow(image10)
image26=outputs2[808,1,:,:].detach().numpy()
#plt.imshow(image11)
image27=outputs1[808,1,:,:].detach().numpy()
#plt.imshow(image12)

arr7[0].imshow(image25)
arr7[0].axis('off')
arr7[1].imshow(image26)
arr7[1].axis('off')
arr7[2].imshow(image27)
arr7[2].axis('off')


# In[291]:


f8, arr8 = plt.subplots(1,3, figsize=(16,5))
image31=images[2012].detach().numpy()
labels[2012]
print (classes[labels[2012]])
image31 = np.transpose(image31, (1, 2, 0))
#plt.imshow(image10)
image32=outputs2[2012,1,:,:].detach().numpy()
#plt.imshow(image11)
image33=outputs1[2012,1,:,:].detach().numpy()
#plt.imshow(image12)

arr8[0].imshow(image31)
arr8[0].axis('off')
arr8[1].imshow(image32)
arr8[1].axis('off')
arr8[2].imshow(image33)
arr8[2].axis('off')


# In[298]:


f9, arr9 = plt.subplots(1,3, figsize=(16,5))
image28=images[1947].detach().numpy()
labels[1947]
print (classes[labels[1947]])
image28= np.transpose(image28, (1, 2, 0))
#plt.imshow(image10)
image29=outputs2[1947,1,:,:].detach().numpy()
#plt.imshow(image11)
image30=outputs1[1947,1,:,:].detach().numpy()
#plt.imshow(image12)

arr9[0].imshow(image28)
arr9[0].axis('off')
arr9[1].imshow(image29)
arr9[1].axis('off')
arr9[2].imshow(image30)
arr9[2].axis('off')


# In[322]:


np.reshape(image1_gray,(32,32))


# In[323]:


np.shape(image1_gray)


# In[363]:


def RGB2gray(image):
    image=color.rgb2gray(image)
    return image
f_hog, arr_hog = plt.subplots(2,10, figsize=(16,5))


image1_gray=RGB2gray(image1)
arr_hog[0,0].imshow(image1_gray, cmap='gray')
arr_hog[0,0].axis('off')
_, hog_vis = extractFeature(image1_gray, vis=True)
arr_hog[1,0].imshow(hog_vis, cmap='gray')
arr_hog[1,0].axis('off')

image4_gray=RGB2gray(image4)
arr_hog[0,1].imshow(image4_gray, cmap='gray')
arr_hog[0,1].axis('off')
_, hog_vis = extractFeature(image4_gray, vis=True)
arr_hog[1,1].imshow(hog_vis, cmap='gray')
arr_hog[1,1].axis('off')

image7_gray=RGB2gray(image7)
arr_hog[0,2].imshow(image7_gray, cmap='gray')
arr_hog[0,2].axis('off')
_, hog_vis = extractFeature(image7_gray, vis=True)
arr_hog[1,2].imshow(hog_vis, cmap='gray')
arr_hog[1,2].axis('off')

image10_gray=RGB2gray(image10)
arr_hog[0,3].imshow(image10_gray, cmap='gray')
arr_hog[0,3].axis('off')
_, hog_vis = extractFeature(image10_gray, vis=True)
arr_hog[1,3].imshow(hog_vis, cmap='gray')
arr_hog[1,3].axis('off')

image13_gray=RGB2gray(image13)
arr_hog[0,4].imshow(image13_gray, cmap='gray')
arr_hog[0,4].axis('off')
_, hog_vis = extractFeature(image13_gray, vis=True)
arr_hog[1,4].imshow(hog_vis, cmap='gray')
arr_hog[1,4].axis('off')

image16_gray=RGB2gray(image16)
arr_hog[0,5].imshow(image16_gray, cmap='gray')
arr_hog[0,5].axis('off')
_, hog_vis = extractFeature(image16_gray, vis=True)
arr_hog[1,5].imshow(hog_vis, cmap='gray')
arr_hog[1,5].axis('off')

image19_gray=RGB2gray(image19)
arr_hog[0,6].imshow(image19_gray, cmap='gray')
arr_hog[0,6].axis('off')
_, hog_vis = extractFeature(image19_gray, vis=True)
arr_hog[1,6].imshow(hog_vis, cmap='gray')
arr_hog[1,6].axis('off')

image22_gray=RGB2gray(image22)
arr_hog[0,7].imshow(image22_gray, cmap='gray')
arr_hog[0,7].axis('off')
_, hog_vis = extractFeature(image22_gray, vis=True)
arr_hog[1,7].imshow(hog_vis, cmap='gray')
arr_hog[1,7].axis('off')

image25_gray=RGB2gray(image25)
arr_hog[0,8].imshow(image25_gray, cmap='gray')
arr_hog[0,8].axis('off')
_, hog_vis = extractFeature(image25_gray, vis=True)
arr_hog[1,8].imshow(hog_vis, cmap='gray')
arr_hog[1,8].axis('off')

image28_gray=RGB2gray(image28)
arr_hog[0,9].imshow(image28_gray, cmap='gray')
arr_hog[0,9].axis('off')
_, hog_vis = extractFeature(image28_gray, vis=True)
arr_hog[1,9].imshow(hog_vis, cmap='gray')
arr_hog[1,9].axis('off')


# In[340]:


plt.imshow(image1_gray, cmap='gray')


# In[342]:


np.shape(hog_vis1)


# In[365]:


hog_vis1.size


# In[368]:


nfeatures = extractFeature(image1_gray, vis=False).size


# In[369]:


nfeatures


# In[547]:


from sklearn.decomposition import PCA
pca=PCA()
pca.fit(test_svm)
X = pca.transform(test_svm)


# In[548]:


np.shape(X)

