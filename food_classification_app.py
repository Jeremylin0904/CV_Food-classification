!pip install anvil-uplink

from google.colab import drive
drive.mount('/content/drive')

import anvil.server

#anvil.server.connect("YOUR OWN KEY")

import tensorflow as tf
import numpy as np

domainselector = tf.keras.models.load_model('model(domainselection).model')
modelw = tf.keras.models.load_model('model(web).model')
modelk = tf.keras.models.load_model('model(Kaggle).model')


weightw = np.array((modelw.layers[482].weights,modelw.layers[483].weights,modelw.layers[484].weights,modelw.layers[485].weights,modelw.layers[486].weights,modelw.layers[487].weights))

weightk = np.array((modelk.layers[482].weights,modelk.layers[483].weights,modelk.layers[484].weights,modelk.layers[485].weights,modelk.layers[486].weights,modelk.layers[487].weights))

#482 to 488
for i in range(482,488):
  def customregularizor(x):
    import tensorflow as tf
    return 0.1 * tf.reduce_sum(tf.square(x-weightw[i]))
  modelw.layers[i].kernel_regularizer = customregularizor

#482 to 488
for i in range(482,488):
  def customregularizor(x):
    import tensorflow as tf
    return 0.1 * tf.reduce_sum(tf.square(x-weightk[i]))
  modelk.layers[i].kernel_regularizer = customregularizor

@anvil.server.callable
def predict(img):
  import pandas as pd
  import tensorflow as tf
  import cv2
  import matplotlib.pyplot as plt
  import numpy as np

  arr = np.fromstring(img.get_bytes(), np.uint8)
  img_array = cv2.imdecode(arr, cv2.IMREAD_COLOR)
  img_array2 = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
  plt.imshow(img_array2)
  plt.show()

  
  
  IMG_SIZE = 200

  new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
  
  domain = domainselector.predict(np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3))

  '''
  Sort predict probility
  @input: list
  @output: list (from high probility to low)
  '''
  def SortPredictprob(data):
    for i in range(len(data)-2):
      for j in range(len(data)-i-1):
        if data[j][0] < data[j+1][0]:
          data[j], data[j+1] = data[j+1], data[j]
    return data

  if domain[0]>0.5:
    

    predictprob = modelw.predict(np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3))

    column_names=["index","name"]
    df = pd.read_csv("tw_food_101_classes.csv", names=column_names)
    foods = df.name.to_list()
    x = []
    for i in range(101):
      x.append([predictprob[0][i],foods[i]])
    
    
    sortedProb = SortPredictprob(x)
    # return sortedProb(x[0:4]) # print top-5 predict result
    d = np.around(domain[0])
    return x[0:5]

  if domain[0]<0.5:
    
    
    predictprob = modelk.predict(np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3))

    column_names=["index","name"]
    df = pd.read_csv("tw_food_101_classes.csv", names=column_names)
    foods = df.name.to_list()
    x = []
    for i in range(101):
      x.append([predictprob[0][i],foods[i]])
    
    
    sortedProb = SortPredictprob(x)
    # return sortedProb(x[0:4]) # print top-5 predict result
    d = np.around(domain[0])        
    return x[0:5]

@anvil.server.callable
def finetune(img, label):
  import pandas as pd
  import tensorflow as tf
  import cv2
  import matplotlib.pyplot as plt
  import numpy as np

  column_names=["index","name"]

  df = pd.read_csv("tw_food_101_classes.csv", names=column_names)

  foods = df.name.to_list()

  y = np.array((foods.index(label),foods.index(label)))

  IMG_SIZE = 200

  arr = np.fromstring(img.get_bytes(), np.uint8)
  img_array = cv2.imdecode(arr, cv2.IMREAD_COLOR)

  x = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
  

    
  IMG_SIZE = 200

  new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    
  domain = domainselector.predict(np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3))

  if domain[0]>0.5:
    

    modelw.fit(np.array((x,x)),y)

    modelw.save('model(web).model')
  if domain[0]<0.5:


    modelk.fit(np.array((x,x)),y)

    modelk.save('model(Kaggle).model')

anvil.server.wait_forever()
