### Here we have the final version, after manual changes

# III.b.1 : Build subsets

## --> Copy-paste the code snippet in the above cell (data preprocessing, train and test data)

# Decentralized structure

for i in skuSet:
  
  X_train_i,X_test_i = np.split(X_dict[i]["train"], [int(0.7 *0.7*98)]) #split for X
  y_train_i,y_test_i = np.split(y_dict[i]["train"], [int(0.7 *0.7*98)]) #split for y 

  X_dict_subsplit[i] = {'train': X_train_i, 'test': X_test_i} #filling dictionary
  y_dict_subsplit[i] = {'train': y_train_i, 'test': y_test_i}

  y_validation += list(y_test_i) #creating the complete training array
  y_subtrain += list(y_train_i) #creating the complete testing array
 
# Centralized structure

X_cen_subtrain = X_dict_subsplit[skuSet[0]]['train'] #initialization with item 0 #X_cen_train_sub
X_cen_validation = X_dict_subsplit[skuSet[0]]['test'] #X_cen_test_sub

for i in skuSet[1:]: #Iteration over items
    X_cen_subtrain = np.concatenate((X_cen_subtrain, X_dict_subsplit[i]['train']), axis = 0) #Bringing together the training set
    X_cen_validation = np.concatenate((X_cen_validation, X_dict_subsplit[i]['test']), axis = 0)

    
# III.1.b.

# b) Run model:

DT_cen = DecisionTreeRegressor(max_features=mf, max_depth=md, random_state=0).fit(X_cen_subtrain, y_subtrain)
score=r2_score(y_validation, DT_cen.predict(X_cen_validation))
print('  R2:',score)

# c) Compare performances on validation data:

if score > maximum_score:
  params = [mf,md]
  maximum_score = score

# III.1.c. --> copy-paste from III.1.a.

DT_cen = DecisionTreeRegressor(max_features=17,max_depth=4,random_state=0).fit(X_cen_train, y_train)
print('OOS R2:',round(r2_score(y_test, DT_cen.predict(X_cen_test)),3))

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

  # b) Run model: --> We add the code for decentralized, that we saw in the first notebook

  y_pred = []
  for i in skuSet:
      model_i = DecisionTreeRegressor(max_features=mf,max_depth=md,random_state=0).fit(X_dict_subsplit[i]['train'] , y_dict_subsplit[i]['train'])
      y_pred += list(model_i.predict(X_dict_subsplit[i]['test']))
  score=r2_score(y_validation, np.array(y_pred))
  print('  R2:',score)

  # c) Compare performances on validation data:
  
  if score > maximum_score:
    params = [mf,md]
    maximum_score = score

print('\nBest Model:')
print('Parameters:',params)
print('Validation R2:',maximum_score)


## Final model ## --> Copy paste from III.1.c., and then change centralized-->decentralized

tZero=time.time()

y_pred = []
for i in skuSet:
    model_i = DecisionTreeRegressor(max_features=mf, max_depth=md, random_state=0).fit(X_dict[i]['train'] , y_dict[i]['train'])
    y_pred += list(model_i.predict(X_dict[i]['test']))

print('OOS R2:',round(r2_score(y_test, np.array(y_pred)),3))

t = time.time()-tZero
print("Time to compute:",round(t,3)," sec.")













