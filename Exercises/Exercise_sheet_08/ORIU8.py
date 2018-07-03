
# coding: utf-8

# # Exercise Sheet 8

# ## Task 0

# In[10]:


import numpy as np
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


def load_data(dset, split='train'):
    '''Get the data and labels of dset.

    Arguments:
        dset: string nameing the dataset. Must be one of
                toy1, toy2, mnist
        split: string defining which split of the data to use. Must be one of
                train, test

    Returns:
        X: np.array containing the data features
        Y: np.array containing the labels for each feature
    '''

    assert dset in ['toy1', 'toy2', 'mnist'],             'dset mus be one of \'toy1\', \'toy2\' or \'catsanddogs\' but '             +'is {}'.format(dset)
    assert split in ['train', 'test'],             'split mus be one of \'train\' or \'test\' but is {}'.format(split)

    if 'toy' in dset:
        X = np.genfromtxt('X_{}_{}.csv'.format(dset, split), dtype=float)
        Y = np.genfromtxt('Y_{}_{}.csv'.format(dset, split), dtype=float)
    else:
        # Load arrays read only to save ram
        trs = [transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))]

        train = split == 'train'
        Mtr = datasets.MNIST('data/',
                             train=train,
                             download=True,
                             transform=transforms.Compose(trs))

        if train:
            X = Mtr.train_data.numpy()
            Y = Mtr.train_labels.numpy()
        else:
            X = Mtr.test_data.numpy()
            Y = Mtr.test_labels.numpy()

    return X, Y


# In[12]:


A,B=load_data('mnist', split='train');
print (A.shape)
print (B.shape)
plt.imshow(A[1], cmap='gray')


# In[22]:


plt.imshow(A[3000], cmap='gray')


# In[15]:


toy1_train_A, toy1_train_B=load_data('toy1', split='train');
print (toy1_train_A.shape)
print (toy1_train_B.shape)
plt.plot(toy1_train_A)


# In[16]:


toy1_test_A, toy1_test_B=load_data('toy1', split='test');
print (toy1_test_A.shape)
print (toy1_test_B.shape)
plt.plot(toy1_test_A)


# In[18]:


toy2_train_A, toy2_train_B=load_data('toy2', split='train');
print (toy2_train_A.shape)
print (toy2_train_B.shape)
plt.plot(toy2_train_A)


# In[19]:


toy2_test_A, toy2_test_B=load_data('toy2', split='test');
print (toy2_test_A.shape)
print (toy2_test_B.shape)
plt.plot(toy2_test_A)


# ## Task 1

# ## 1.

# In[83]:


from sklearn.svm import SVC

def classifier_fn(s):
    
    
    clf = SVC()
    clf.fit(toy1_train_A,toy1_train_B) 
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    return clf.predict(s)




# In[84]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_2d_data(samples, classifier_fn, density=3, fixed_bounds=None, title='Please add a title!'):
    '''Shows the bounds of a classifier and some data on top of it.
   
    Arguments:
        samples: list of 2d data arrays
        classifier: function that returns the class of a set of sample points
        density: Number of datapoints per unit used for generating 
        the bounds map
        fixed_bounds: [minx, maxx, miny, maxy] array defining the
        area in which the classifier is evaluated
   
    Returns:
        f: plt.figure instance
        ax: plt.axes instance
        
     '''
    
    assert isinstance(samples, list),            'Please supply your data as list of 2d numpy arrays containing '            'samples for each class.'
    
    assert np.all([s.ndim == 2 for s in samples]),             'Samples must be a list of 2d arrays, but got a list of '             'arrays of dimensions {}'.format([s.ndim for s in samples])
    
    all_samples = np.concatenate(samples, axis=0)

    if fixed_bounds is None:
       # Determine the bounds of the data
       mins = np.min(all_samples, axis=0)
       maxs = np.max(all_samples, axis=0)
       ranges = maxs - mins
       mins = mins - 0.1 * ranges
       maxs = maxs + 0.1 * ranges
       minx, miny = mins
       maxx, maxy = maxs
    else:
       minx, maxx, miny, maxy = fixed_bounds
    
    # Generate a per class colormap showing the bounds of classifier
    samples_x = int((maxx-minx)*density)
    samples_y = int((maxy-miny)*density)
    
    bound_map = np.zeros([samples_x, samples_y])
    samplings_x = np.linspace(minx, maxx, num=samples_x)
    samplings_y = np.linspace(miny, maxy, num=samples_y)
    
    for i, x in enumerate(samplings_x):
       for j, y in enumerate(samplings_y):
           class_label = classifier_fn([[x, y]])
           bound_map[i, j] = int(class_label)
    
    N_classes = len(np.unique(bound_map))
    
    # Define matching colormap
    cmap = plt.cm.viridis
    # extract all colors from the .jet map
    cmap = colors.ListedColormap([cmap(i) for i in range(cmap.N)])
    bounds = np.linspace(-0.5, N_classes-0.5, N_classes+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Do the actual plotting
    f, ax = plt.subplots(1,1)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    for sample in samples:
       ax.scatter(sample[:, 0], sample[:, 1])
    im = ax.imshow(bound_map.T,
    extent=[minx, maxx, maxy, miny],
    cmap=cmap,
    norm=norm)
    f.colorbar(im, cmap=cmap, norm=norm, ticks=range(N_classes), label='class') 
    ax.set_aspect(1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title is not None:
       ax.set_title(title)
    return f, ax


# In[86]:


N_classes = len(np.unique(toy1_train_B))
samples = []
for i in range(N_classes):
        samples.append(np.array(toy1_train_A[toy1_train_B==i]))
plot_2d_data(samples, classifier_fn, density=3, fixed_bounds=None, title='Toy 1 training data')


# In[87]:


from sklearn.svm import SVC

def classifier_fn(s):
    
    
    clf = SVC()
    clf.fit(toy2_train_A,toy2_train_B) 
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    return clf.predict(s)


# In[88]:


N_classes = len(np.unique(toy2_train_B))
samples = []
for i in range(N_classes):
        samples.append(np.array(toy2_train_A[toy2_train_B==i]))
plot_2d_data(samples, classifier_fn, density=3, fixed_bounds=None, title='Toy 2 training data')


# ## 2

# In[99]:


from sklearn.metrics import accuracy_score
clf = SVC()
clf.fit(toy2_train_A,toy2_train_B) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',    max_iter=-1, probability=False, random_state=None, shrinking=True,
tol=0.001, verbose=False)
predicted=clf.predict(toy2_test_A)
print (accuracy_score(toy2_test_B, predicted))


# In[100]:


from sklearn.metrics import accuracy_score
clf = SVC()
clf.fit(toy1_train_A,toy1_train_B) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',    max_iter=-1, probability=False, random_state=None, shrinking=True,
tol=0.001, verbose=False)
predicted=clf.predict(toy1_test_A)
print (accuracy_score(toy1_test_B, predicted))


# REASONS FOR THE RESULT:
# 

# ## 3

# Kernel Trick

# ## 4

# ## Task 2: Toy data
# ## 1.

# In[105]:


from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d
from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

try:
    # Import tqdm to get some nice loading bars.
    # If you don't have it I would highly recommend it:
    # pip install tqdm
    from tqdm import tqdm, trange
    print_fn = tqdm.write
    has_tqdm = True
except ImportError:
    print_fn = print
    has_tqdm = False




# Don't worry about this:
def initializer(module):
    '''This function can be used to set the values of the weight and bias
    tensors of `module`.
    '''
    if isinstance(module, Conv2d):
        xavier_normal_(module.weight)
        normal_(module.bias)
    elif isinstance(module, Linear):
        normal_(module.weight)
        normal_(module.bias)


class FullyConnected(nn.Module):
    '''Please fill in the blanks.
    This Model is trained using the train function.
    '''
    def __init__(self, D_in, H, D_out):
        '''Create all model parts here.
        Make sure that all parts are attributes of this object.

        Arguments:
            n_classes: int specifieing the output dimension of the model.
                    Use this in your model definition.
        '''
        super(FullyConnected, self).__init__()

        self.fc1 = Linear(D_in, H)
         
        self.middle_linear1 = Linear(H, H)
        self.middle_linear2 = Linear(H, H)
        self.output_linear = Linear(H, D_out)
        
        self.activation = ReLU()

        # this part fills in values of all weight and bias tensors of the model
        self.apply(initializer)

    def forward(self, x):
        '''Apply the model to some data x.

        Arguments:
            x: 2d torch Tensor

        Returns:
            out: class prediction
        '''
        x = self.fc1(x)
        x = self.activation(x)
        x = self.middle_linear1(x)
        x = self.activation(x)
        x = self.middle_linear1(x)
        x = self.activation(x)
        out=self.output_linear1(x)
        
        return out






# ## 2

# In[ ]:


def train(model, dataloader, n_epochs=10, checkpoint_name='training',
          use_gpu=True):
    '''Use this function to trein the above two models.

    Arguments:
        model: nn.Model instance, e.g. a `MyConvNet` or a
                `FullyConneted` instance.
        dataloader: torch.util.data.Dataloader instance, using e.g a MyDataSet
                instance. See the example code below.
        n_epochs: int specifying how many times to iterate over the dataset.
        checkpoint_name: string used to name the model checkpoint saved
                every epoch. It contains all model weights.
        use_gpu: bool, push tensors and model etc to the gpu
    '''

    if use_gpu:
        model.cuda()

    Loss = nn.CrossEntropyLoss()
    Optimizer = optim.Adam(model.parameters(), lr = 0.0001) 

    if has_tqdm:
        e_iterator = trange(n_epochs, desc='epoch')
    else:
        e_iterator = range(n_epochs)

    for e in e_iterator:

        if has_tqdm:
            iterator = enumerate(tqdm(dataloader, desc='batch'))
        else:
            iterator = enumerate(dataloader)

        for e_step, (x, y) in iterator:
            train_step = e_step + len(dataloader)*e

            if use_gpu:
                x = x.cuda()
                y = y.cuda()

            # Forward
            # Your code here
            prediction = model(x)

            # Loss
            # Your code here
            loss = Loss(prediction, y)
            acc = torch.mean(torch.eq(torch.argmax(prediction, dim=-1),
                                      y).float())

            Optimizer.zero_grad()

            # Backward
            loss.backward()
            
            # Update
            Optimizer.step()

            if train_step % 25 == 0:
                print_fn('{}: Batch-Accuracy = {}, Loss = {}'                          .format(train_step, float(acc), float(loss)))
        torch.save(model.state_dict(), '{}-{}.ckpt'.format(checkpoint_name, e))

