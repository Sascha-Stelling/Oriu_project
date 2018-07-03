
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


# In[85]:


N_classes = len(np.unique(toy1_train_B))
samples = []
for i in range(N_classes):
        samples.append(np.array(toy1_train_A[toy1_train_B==i]))
plot_2d_data(samples, classifier_fn, density=3, fixed_bounds=None, title='Please add a title!')


# In[79]:


samples

