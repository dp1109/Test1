import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
     


# In[25]:


# loading the diabetes dataset to a pandas DataFrame
insta_dataset = pd.read_csv(r"C:\Users\USER\Downloads\train.csv")


# In[26]:


# getting the statistical measures of the data
insta_dataset.describe()


# In[27]:


insta_dataset['fake'].value_counts()


# In[28]:


insta_dataset.groupby('fake').mean()


# In[29]:


# separating the data and labels
X = insta_dataset.drop(columns = 'fake', axis=1)
Y = insta_dataset['fake']


# In[30]:


scaler = StandardScaler()


# In[31]:


scaler.fit(X)


# In[32]:


standardized_data = scaler.transform(X)


# In[33]:


X = standardized_data
Y = insta_dataset['fake']


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.8, stratify=Y, random_state=2)


# In[35]:


classifier = svm.SVC(kernel='linear')


# In[36]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[37]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[38]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[39]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[40]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[41]:


input_data = (0,0,1,0,0,0,0,0,0,17,44)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person the parson Instagram id is not fake')
elif (prediction[0] == 1):
  print('The person the parson Instagram id is fake')


