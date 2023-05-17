### Here we have the code to copy paste


# III. Tree-based methods

# III.b.1 : Build subsets

## --> Copy-paste the code snippet in the above cell (data preprocessing, train and test data)

# Decentralized structure

for i in skuSet:
  
  X_train_i,X_test_i = np.split(skuData[i]["X"], [int(0.70 *98)]) #split for X
  y_train_i,y_test_i = np.split(skuData[i]["y"], [int(0.70 *98)]) #split for y 

  X_dict[i] = {'train': X_train_i, 'test': X_test_i} #filling dictionary
  y_dict[i] = {'train': y_train_i, 'test': y_test_i}

  y_test += list(y_test_i) 
  y_train += list(y_train_i) 
 
# Centralized structure

X_cen_train = X_dict[skuSet[0]]['train'] #initialization with item 0
X_cen_test = X_dict[skuSet[0]]['test']

for i in skuSet[1:]: #Iteration over items
    X_cen_train = np.concatenate((X_cen_train, X_dict[i]['train']), axis = 0) #Bringing together the training set
    X_cen_test = np.concatenate((X_cen_test, X_dict[i]['test']), axis = 0)

    
# III.2.

# b) Run model: --> copy-paste from III.1.a.

DT_cen_visualization = DecisionTreeRegressor(max_features=43,max_depth=3,random_state=0).fit(X_cen_train, y_train)
print('OOS R2:',round(r2_score(y_test, DT_cen_visualization.predict(X_cen_test)),3))

# c) Compare performances on validation data:

#write manually



# III.2.c. --> copy-paste from III.1.a.

DT_cen_visualization = DecisionTreeRegressor(max_features=43,max_depth=3,random_state=0).fit(X_cen_train, y_train)
print('OOS R2:',round(r2_score(y_test, DT_cen_visualization.predict(X_cen_test)),3))

# III.2. Breakout rooms


## Select optimal parameters ## --> Copy paste from III.1.b.

max_features_ = list(range(2,45)) 
max_depth_ = list(range(2,10))
params=[]
maximum_score=0

#selection of parameters to test
random.seed(5)
mf_ = random.choices(max_features_, k=50)
md_ = random.choices(max_depth_, k=50)

## Iterations to select best model
for i in range (50):
  print('Model number:',i+1)

  # a) Selection of parameters to test
  mf = mf_[i]
  md = md_[i]
  print('  Parameters:',[mf,md])

  # b) Run model:

  DT_cen = DecisionTreeRegressor(max_features=mf, max_depth=md, random_state=0).fit(X_cen_subtrain, y_subtrain)
  score=r2_score(y_validation, DT_cen.predict(X_cen_validation))
  
  
  
  print('  R2:',score)

  # c) Compare performances on validation data:
  
  if score > maximum_score:
    params = [mf,md]
    maximum_score = score

print('\nBest Model:')
print('Parameters:',params)
print('Validation R2:',maximum_score)


## Final model ## --> Copy paste from III.1.c.

tZero=time.time()

mf,md=params
DT_cen = DecisionTreeRegressor(max_features=17,max_depth=4,random_state=0).fit(X_cen_train, y_train)



print('OOS R2:',round(r2_score(y_test, DT_cen.predict(X_cen_test)),3))

t = time.time()-tZero
print("Time to compute:",round(t,3)," sec")



# IV. Clustering methods

# IV.1. --> nothing to add


# IV.2.

## IV.2.a. Loop to find the optimal number of clusters


#Clustering --> copy-paste from IV.1.a.

d = len(colnames) #d is the number of columns
X_clus = np.zeros((len(skuSet), d))

for sku in skuSet:
    sku_index=sku-1
    X_clus[sku_index, :] = np.mean(X_dict[sku]['train'], axis = 0)
X_clus

kmeans = KMeans(n_clusters=z, random_state=0).fit(X_clus)


#Build dataset and run model --> copy-paste from IV.1.c. (Loop with all clusters)

#Loop
y_clus_pred = []
y_clus_test = []

for j in range(z):

  ## 1. Get indices of items in cluster j 
  clus_items = list(np.where(kmeans.labels_ == j)[0])

  ## 2. Build dataset

  ##Initialization 
  #X
  X_clus_j_train = X_dict[skuSet[clus_items[0]]]['train'] #initialization with first item of the cluster
  X_clus_j_test = X_dict[skuSet[clus_items[0]]]['test']
  #y
  y_clus_j_train = list(y_dict[skuSet[clus_items[0]]]['train']) #initialization with first item of the cluster
  y_clus_j_test = list(y_dict[skuSet[clus_items[0]]]['test'])
  ##Loop 
  for idx in clus_items[1:]: #Iteration over items
    sku=skuSet[idx]
    #X
    X_clus_j_train = np.concatenate((X_clus_j_train, X_dict[sku]['train']), axis = 0) #Bringing together the training set for the cluster
    X_clus_j_test = np.concatenate((X_clus_j_test, X_dict[sku]['test']), axis = 0)
    #y
    y_clus_j_train += list(y_dict[sku]['train'])
    y_clus_j_test += list(y_dict[sku]['test'])

  ## 3. Run model
  model_clus_j = LinearRegression().fit(X_clus_j_train, y_clus_j_train)
  y_clus_pred += list(model_clus_j.predict(X_clus_j_test))
  y_clus_test += y_clus_j_test


#Comparison of results --> write manually







## At the very end of the loop (no indentation) --> write manually









## IV.2.a. Loop to find the optimal number of clusters

# Clustering --> copy-paste from IV.1.a.

d = len(colnames) #d is the number of columns
X_clus = np.zeros((len(skuSet), d))

for sku in skuSet:
    sku_index=sku-1
    X_clus[sku_index, :] = np.mean(X_dict[sku]['train'], axis = 0)
X_clus

kmeans = KMeans(n_clusters=z, random_state=0).fit(X_clus)

# Build dataset and run model for each cluster --> copy-paste from IV.1.c. (Loop with all clusters)

#Loop
y_clus_pred = []
y_clus_test = []

for j in range(z):

  ## 1. Get indices of items in cluster j 
  clus_items = list(np.where(kmeans.labels_ == j)[0])

  ## 2. Build dataset

  ##Initialization 
  #X
  X_clus_j_train = X_dict[skuSet[clus_items[0]]]['train'] #initialization with first item of the cluster
  X_clus_j_test = X_dict[skuSet[clus_items[0]]]['test']
  #y
  y_clus_j_train = list(y_dict[skuSet[clus_items[0]]]['train']) #initialization with first item of the cluster
  y_clus_j_test = list(y_dict[skuSet[clus_items[0]]]['test'])
  ##Loop 
  for idx in clus_items[1:]: #Iteration over items
    sku=skuSet[idx]
    #X
    X_clus_j_train = np.concatenate((X_clus_j_train, X_dict[sku]['train']), axis = 0) #Bringing together the training set for the cluster
    X_clus_j_test = np.concatenate((X_clus_j_test, X_dict[sku]['test']), axis = 0)
    #y
    y_clus_j_train += list(y_dict[sku]['train'])
    y_clus_j_test += list(y_dict[sku]['test'])

  ## 3. Run model
  model_clus_j = LinearRegression().fit(X_clus_j_train, y_clus_j_train)
  y_clus_pred += list(model_clus_j.predict(X_clus_j_test))
  y_clus_test += y_clus_j_test

  
#### Print Results #### --> write manually









    
    
    
    
 
