{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2fb93505",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'tensorflow'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn [1], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtensorflow\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mtf\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mmatplotlib\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m pyplot \u001b[38;5;28;01mas\u001b[39;00m plt\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtensorflow\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mkeras\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Model\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'tensorflow'"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from matplotlib import pyplot as plt\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.preprocessing import image\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Conv2D,Dropout,Flatten,MaxPooling2D\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "dcdd28c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_path=\"/home/fansan/Desktop/Document-Classifier/Dataset/data/train/\"\n",
    "test_path=\"/home/fansan/Desktop/Document-Classifier/Dataset/data/test/\"\n",
    "val_path=\"/home/fansan/Desktop/Document-Classifier/Dataset/data/valid/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "08737e69",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train=[]\n",
    "for folder in os.listdir(train_path):\n",
    "    sub_path=train_path+\"/\"+folder\n",
    "    for img in os.listdir(sub_path):\n",
    "        image_path=sub_path+\"/\"+img\n",
    "        img_arr=cv2.imread(image_path)\n",
    "        img_arr=cv2.resize(img_arr,(224,224))\n",
    "        x_train.append(img_arr)\n",
    "\n",
    "x_test=[]\n",
    "for folder in os.listdir(test_path):\n",
    "    sub_path=test_path+\"/\"+folder\n",
    "    for img in os.listdir(sub_path):\n",
    "        image_path=sub_path+\"/\"+img\n",
    "        img_arr=cv2.imread(image_path)\n",
    "        img_arr=cv2.resize(img_arr,(224,224))\n",
    "        x_test.append(img_arr)\n",
    "\n",
    "x_val=[]\n",
    "for folder in os.listdir(val_path):\n",
    "    sub_path=val_path+\"/\"+folder\n",
    "    for img in os.listdir(sub_path):\n",
    "        image_path=sub_path+\"/\"+img\n",
    "        img_arr=cv2.imread(image_path)\n",
    "        img_arr=cv2.resize(img_arr,(224,224))\n",
    "        x_val.append(img_arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ab988607",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_x=np.array(x_train)\n",
    "test_x=np.array(x_test)\n",
    "val_x=np.array(x_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3afb78f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_x=train_x/255.0\n",
    "test_x=test_x/255.0\n",
    "val_x=val_x/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "cf3cc2ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen = ImageDataGenerator(rescale = 1./255)\n",
    "test_datagen = ImageDataGenerator(rescale = 1./255)\n",
    "val_datagen = ImageDataGenerator(rescale = 1./255)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4818f4b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 145 images belonging to 5 classes.\n",
      "Found 46 images belonging to 5 classes.\n",
      "Found 29 images belonging to 5 classes.\n"
     ]
    }
   ],
   "source": [
    "training_set = train_datagen.flow_from_directory(train_path,\n",
    "                                                 target_size = (224, 224),\n",
    "                                                 batch_size = 32,\n",
    "                                                 class_mode = 'sparse')\n",
    "test_set = test_datagen.flow_from_directory(test_path,\n",
    "                                            target_size = (224, 224),\n",
    "                                            batch_size = 32,\n",
    "                                            class_mode = 'sparse')\n",
    "val_set = val_datagen.flow_from_directory(val_path,\n",
    "                                            target_size = (224, 224),\n",
    "                                            batch_size = 32,\n",
    "                                            class_mode = 'sparse')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a4c35ec2",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_y=training_set.classes\n",
    "test_y=test_set.classes\n",
    "val_y=val_set.classes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1b939cab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'aadhar': 0, 'driver license': 1, 'pan': 2, 'passport': 3, 'voter': 4}"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_set.class_indices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f879aed5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((145,), (46,), (29,))"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_y.shape,test_y.shape,val_y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d05e542a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-02-02 17:49:35.237771: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/fansan/anaconda3/lib/python3.9/site-packages/cv2/../../lib64:\n",
      "2023-02-02 17:49:35.238147: W tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: UNKNOWN ERROR (303)\n",
      "2023-02-02 17:49:35.238196: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fansan-HP-Laptop-15s-du1xxx): /proc/driver/nvidia/version does not exist\n",
      "2023-02-02 17:49:35.239836: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
      "2023-02-02 17:49:35.313035: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 403734528 exceeds 10% of free system memory.\n",
      "2023-02-02 17:49:35.556831: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 403734528 exceeds 10% of free system memory.\n",
      "2023-02-02 17:49:35.809072: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 403734528 exceeds 10% of free system memory.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d (Conv2D)             (None, 222, 222, 32)      896       \n",
      "                                                                 \n",
      " max_pooling2d (MaxPooling2D  (None, 111, 111, 32)     0         \n",
      " )                                                               \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 394272)            0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 256)               100933888 \n",
      "                                                                 \n",
      " dropout (Dropout)           (None, 256)               0         \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 5)                 1285      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 100,936,069\n",
      "Trainable params: 100,936,069\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "input_shape=(224,224, 3)\n",
    "model=Sequential()\n",
    "model.add(Conv2D(32,kernel_size=(3,3),input_shape=input_shape))\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(256,activation=tf.nn.relu))\n",
    "model.add(Dropout(0.1))\n",
    "model.add(Dense(5,activation=tf.nn.softmax))\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5ee9b644",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(\n",
    "  loss='sparse_categorical_crossentropy',\n",
    "  optimizer=\"adam\",\n",
    "  metrics=['accuracy']\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9291ccff",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5a3769ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-02-02 17:49:37.182283: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 87306240 exceeds 10% of free system memory.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-02-02 17:49:37.578295: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 403734528 exceeds 10% of free system memory.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 16s 3s/step - loss: 141.0377 - accuracy: 0.2276 - val_loss: 124.2629 - val_accuracy: 0.2414\n",
      "Epoch 2/20\n",
      "5/5 [==============================] - 12s 2s/step - loss: 53.9360 - accuracy: 0.4690 - val_loss: 21.6733 - val_accuracy: 0.1379\n",
      "Epoch 3/20\n",
      "5/5 [==============================] - 17s 3s/step - loss: 15.6712 - accuracy: 0.3724 - val_loss: 14.4903 - val_accuracy: 0.2414\n",
      "Epoch 4/20\n",
      "5/5 [==============================] - 12s 2s/step - loss: 6.7062 - accuracy: 0.4828 - val_loss: 7.8116 - val_accuracy: 0.1379\n",
      "Epoch 5/20\n",
      "5/5 [==============================] - 21s 4s/step - loss: 3.3709 - accuracy: 0.5448 - val_loss: 7.5263 - val_accuracy: 0.2414\n",
      "Epoch 6/20\n",
      "5/5 [==============================] - 21s 4s/step - loss: 2.2906 - accuracy: 0.5862 - val_loss: 3.8181 - val_accuracy: 0.2069\n",
      "Epoch 7/20\n",
      "5/5 [==============================] - 16s 2s/step - loss: 0.8676 - accuracy: 0.7172 - val_loss: 3.1424 - val_accuracy: 0.4138\n",
      "Epoch 8/20\n",
      "5/5 [==============================] - 13s 3s/step - loss: 0.5694 - accuracy: 0.8000 - val_loss: 2.6040 - val_accuracy: 0.3793\n",
      "Epoch 9/20\n",
      "5/5 [==============================] - 17s 3s/step - loss: 0.5082 - accuracy: 0.7931 - val_loss: 2.2576 - val_accuracy: 0.4483\n",
      "Epoch 10/20\n",
      "5/5 [==============================] - 12s 2s/step - loss: 0.3923 - accuracy: 0.8414 - val_loss: 2.2998 - val_accuracy: 0.2414\n",
      "Epoch 11/20\n",
      "5/5 [==============================] - 17s 3s/step - loss: 0.3517 - accuracy: 0.8828 - val_loss: 2.5929 - val_accuracy: 0.3793\n",
      "Epoch 12/20\n",
      "5/5 [==============================] - 14s 2s/step - loss: 0.3115 - accuracy: 0.8759 - val_loss: 2.3279 - val_accuracy: 0.3448\n",
      "Epoch 13/20\n",
      "5/5 [==============================] - 16s 3s/step - loss: 0.3435 - accuracy: 0.8690 - val_loss: 2.5589 - val_accuracy: 0.3448\n",
      "Epoch 14/20\n",
      "5/5 [==============================] - 16s 3s/step - loss: 0.2710 - accuracy: 0.9103 - val_loss: 2.5613 - val_accuracy: 0.3448\n",
      "Epoch 14: early stopping\n"
     ]
    }
   ],
   "source": [
    "# fit the model\n",
    "history = model.fit(\n",
    "  np.array(train_x),\n",
    "  np.array(train_y),\n",
    "  validation_data=(val_x,val_y),\n",
    "  epochs=20,\n",
    "  callbacks=[early_stop],\n",
    "  batch_size=32,shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e9aaaf5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# accuracies\n",
    "plt.plot(history.history['accuracy'], label='train acc')\n",
    "plt.plot(history.history['val_accuracy'], label='val acc')\n",
    "plt.legend()\n",
    "plt.savefig('acc1.png')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b9e36f21",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAigAAAGdCAYAAAA44ojeAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAABKg0lEQVR4nO3deXyU5b338c/MJJksZIckRBIIEhcWURNqD6hgRRQVF+pWcOHRnscF1BRFpdYWfVWoaBELFZfjox4pR+sRLHU5EKqyHGplERdQFg0JWwhLyJ5MkrmfPyYzySSTZJLMBvm+X695zcx933PPlQlmvl739bsuk2EYBiIiIiIhxBzsBoiIiIi0poAiIiIiIUcBRUREREKOAoqIiIiEHAUUERERCTkKKCIiIhJyFFBEREQk5CigiIiISMgJC3YDusNut3Pw4EFiY2MxmUzBbo6IiIh4wTAMKioqSE9Px2zuuI/kpAwoBw8eJCMjI9jNEBERkW7Yt28fAwYM6PCYkzKgxMbGAo4fMC4uLsitEREREW+Ul5eTkZHh+h7vyEkZUJyXdeLi4hRQRERETjLeDM/QIFkREREJOQooIiIiEnIUUERERCTknJRjUERE5NRmGAYNDQ00NjYGuynSReHh4Vgslh6fRwFFRERCis1m49ChQ1RXVwe7KdINJpOJAQMG0KdPnx6dRwFFRERCht1up6CgAIvFQnp6OhEREZqQ8yRiGAZHjhxh//79ZGdn96gnRQFFRERChs1mw263k5GRQXR0dLCbI93Qr18/9u7dS319fY8CigbJiohIyOlsGnQJXb7q8dK/ABEREQk5CigiIiISchRQREREQsygQYNYuHBh0M8RTBokKyIi0kPjxo3j3HPP9Vkg2LRpEzExMT4518mqyz0o69atY9KkSaSnp2MymXj//ffbPfbuu+/GZDK1+YXV1dVx//3307dvX2JiYrjmmmvYv39/V5vicwdP1PDcqp3M++i7YDdFREROMc7J57zRr1+/Xl/F1OWAUlVVxciRI1m8eHGHx73//vv861//Ij09vc2+vLw8VqxYwdtvv82GDRuorKzk6quvDvqMgdW2BhZ/uoelnxdiGEZQ2yIiIo4v9WpbQ1Bu3n4PTJs2jbVr1/LCCy9gMpkwmUzs3buXzz77DJPJxKpVq8jNzcVqtbJ+/Xp++OEHrr32WlJTU+nTpw+jRo1izZo1budsfXnGZDLxH//xH1x//fVER0eTnZ3NypUru/RZFhUVce2119KnTx/i4uK46aabOHz4sGv/V199xSWXXEJsbCxxcXHk5OSwefNmAAoLC5k0aRKJiYnExMQwbNgwPvrooy69f1d1+RLPxIkTmThxYofHHDhwgBkzZrBq1Squuuoqt31lZWW89tprvPXWW4wfPx6ApUuXkpGRwZo1a7j88su72iSfGZgcQ7jFRJWtkQMnahiQ2LvTq4hIsNXUNzL0t6uC8t47nrqc6IjOvyZfeOEFdu3axfDhw3nqqaeA5rlAAB555BGee+45Bg8eTEJCAvv37+fKK6/k97//PZGRkbz55ptMmjSJnTt3kpmZ2e77PPnkk8yfP59nn32WRYsWMXXqVAoLC0lKSuq0jYZhcN111xETE8PatWtpaGjgvvvu4+abb+azzz4DYOrUqZx33nksWbIEi8XCtm3bCA8PB2D69OnYbDbWrVtHTEwMO3bs6PFMsZ3x+RgUu93ObbfdxqxZsxg2bFib/Vu2bKG+vp4JEya4tqWnpzN8+HA2btzoMaDU1dVRV1fnel5eXu7rZgMQbjGT1TeGXYcr2V1SqYAiIiKdio+PJyIigujoaNLS0trsf+qpp7jssstcz5OTkxk5cqTr+e9//3tWrFjBypUrmTFjRrvvM23aNH7xi18AMHfuXBYtWsQXX3zBFVdc0Wkb16xZw9dff01BQQEZGRkAvPXWWwwbNoxNmzYxatQoioqKmDVrFmeddRYA2dnZrtcXFRXx85//nBEjRgAwePDgTt+zp3weUJ555hnCwsJ44IEHPO4vLi4mIiKCxMREt+2pqakUFxd7fM28efN48sknfd1Uj7JTYx0B5XAFl5yZEpD3FBERz6LCLex4Kjg961HhPV/wDiA3N9fteVVVFU8++SQffPABBw8epKGhgZqaGoqKijo8zznnnON6HBMTQ2xsLCUlJV614bvvviMjI8MVTgCGDh1KQkIC3333HaNGjWLmzJn88pe/dF3huPHGGzn99NMBeOCBB7j33ntZvXo148eP5+c//7lbe/zBp2XGW7Zs4YUXXuCNN97o8kxyhmG0+5rZs2dTVlbmuu3bt88XzfXojJRYAHYdrvTbe4iIiHdMJhPREWFBuflqRtTW1TizZs3ivffe4+mnn2b9+vVs27aNESNGYLPZOjyP83JLy8/Gbrd71Yb2vmNbbp8zZw7bt2/nqquu4pNPPmHo0KGsWLECgF/+8pf8+OOP3HbbbXzzzTfk5uayaNEir967u3waUNavX09JSQmZmZmEhYURFhZGYWEhDz30EIMGDQIgLS0Nm81GaWmp22tLSkpITU31eF6r1UpcXJzbzV+yUx3X1HYfrvDbe4iIyKklIiLC60KP9evXM23aNK6//npGjBhBWlqaa7yKvwwdOpSioiK3/8HfsWMHZWVlnH322a5tZ5xxBr/61a9YvXo1kydP5vXXX3fty8jI4J577mH58uU89NBDvPrqq35ts08Dym233cbXX3/Ntm3bXLf09HRmzZrFqlWOQU45OTmEh4eTn5/vet2hQ4f49ttvGT16tC+b0y1nOANKSaUqeURExCuDBg3iX//6F3v37uXo0aMd9mwMGTKE5cuXs23bNr766iumTJnidU9Id40fP55zzjmHqVOnsnXrVr744gtuv/12xo4dS25uLjU1NcyYMYPPPvuMwsJC/vd//5dNmza5wkteXh6rVq2ioKCArVu38sknn7gFG3/o8hiUyspK9uzZ43peUFDAtm3bSEpKIjMzk+TkZLfjw8PDSUtL48wzzwQcg4nuuusuHnroIZKTk0lKSuLhhx9mxIgRrqqeYHJW8lSrkkdERLz08MMPc8cddzB06FBqamooKCho99jnn3+eO++8k9GjR9O3b18effRRvxV/ODnnLbv//vu5+OKLMZvNXHHFFa7LNBaLhWPHjnH77bdz+PBh+vbty+TJk13jPxsbG5k+fTr79+8nLi6OK664gueff96/bTa62E3w2Wefcckll7TZfscdd/DGG2+02T5o0CDy8vLIy8tzbautrWXWrFksW7aMmpoaLr30Ul588UW3wTsdKS8vJz4+nrKyMr9c7pnw/Fp2Ha7k9WmjuOQsDZQVEQmU2tpaCgoKyMrKIjIyMtjNkW7o6HfYle/vLvegjBs3rkuXPjxdV4uMjGTRokV+H2DTXa5KnpIKBRQREZEg0GKBHqiSR0REJLgUUDxQJY+IiEhwKaB40LKSx25XJY+IiEigKaB40LKS52BZTbCbIyIi0usooHgQbjEzuK/zMo/GoYiIiASaAko7hjRd5tmlcSgiIiIBp4DSDmclz+4S9aCIiIgEmgJKO85QJY+IiATQoEGDWLhwYbv7p02bxnXXXRew9gSbAko7slXJIyIiEjQKKO1ovSaPiIiIBI4CSjtaVvLs0TgUERFpx8svv8xpp53WZkXia665hjvuuAOAH374gWuvvZbU1FT69OnDqFGjWLNmTY/et66ujgceeICUlBQiIyO58MIL2bRpk2t/aWkpU6dOpV+/fkRFRZGdnc3rr78OgM1mY8aMGfTv35/IyEgGDRrEvHnzetQeX1NA6UC2KnlERILLMMBWFZybl+vO3XjjjRw9epRPP/3Uta20tJRVq1YxdepUACorK7nyyitZs2YNX375JZdffjmTJk2iqKio2x/NI488wnvvvcebb77J1q1bGTJkCJdffjnHjx8H4IknnmDHjh18/PHHfPfddyxZsoS+ffsC8Kc//YmVK1fy17/+lZ07d7J06VIGDRrU7bb4Q5cXC+xNslNigUNak0dEJFjqq2FuenDe+9cHISKm08OSkpK44oorWLZsGZdeeikA7777LklJSa7nI0eOZOTIka7X/P73v2fFihWsXLmSGTNmdLlpVVVVLFmyhDfeeIOJEycC8Oqrr5Kfn89rr73GrFmzKCoq4rzzziM3NxfALYAUFRWRnZ3NhRdeiMlkYuDAgV1ug7+pB6UDzkqePSXqQRERkfZNnTqV9957j7q6OgD+8pe/cMstt2CxWABHoHjkkUcYOnQoCQkJ9OnTh++//77bPSg//PAD9fX1jBkzxrUtPDycn/zkJ3z33XcA3Hvvvbz99tuce+65PPLII2zcuNF17LRp09i2bRtnnnkmDzzwAKtXr+7uj+436kHpQHZq81wodruB2WwKcotERHqZ8GhHT0aw3ttLkyZNwm638+GHHzJq1CjWr1/PggULXPtnzZrFqlWreO655xgyZAhRUVHccMMN2Gy2bjXNaLr8ZDKZ2mx3bps4cSKFhYV8+OGHrFmzhksvvZTp06fz3HPPcf7551NQUMDHH3/MmjVruOmmmxg/fjz//d//3a32+IMCSgcGJke7VfJkJHn/j1VERHzAZPLqMkuwRUVFMXnyZP7yl7+wZ88ezjjjDHJyclz7169fz7Rp07j++usBx5iUvXv3dvv9hgwZQkREBBs2bGDKlCkA1NfXs3nzZvLy8lzH9evXj2nTpjFt2jQuuugiZs2axXPPPQdAXFwcN998MzfffDM33HADV1xxBcePHycpKanb7fIlBZQOOCt5dh6uYHdJhQKKiIi0a+rUqUyaNInt27dz6623uu0bMmQIy5cvZ9KkSZhMJp544ok2VT9dERMTw7333susWbNISkoiMzOT+fPnU11dzV133QXAb3/7W3Jychg2bBh1dXV88MEHnH322QA8//zz9O/fn3PPPRez2cy7775LWloaCQkJ3W6TrymgdCI7tSmgHK7kZ2elBrs5IiISon72s5+RlJTEzp07Xb0aTs8//zx33nkno0ePpm/fvjz66KOUl5f36P3+8Ic/YLfbue2226ioqCA3N5dVq1aRmJgIQEREBLNnz2bv3r1ERUVx0UUX8fbbbwPQp08fnnnmGXbv3o3FYmHUqFF89NFHmM2hMzTVZBhe1lGFkPLycuLj4ykrKyMuLs6v7/XCmt08v2YXPz9/AH+8aWTnLxARkW6rra2loKCArKwsIiMjg90c6YaOfodd+f4OnagUolxr8qiSR0REJGAUUDrhrOTZozV5REREAkYBpRODWlXyiIiIiP8poHQirMWaPLrMIyIiEhgKKF5wrsmzW1Pei4iIBIQCihfOaBqHojV5REQC4yQsMJUmvvrdKaB4ITtFl3hERAIhPDwcgOrq6iC3RLrLOX2/cx2i7tJEbV5wrclzWGvyiIj4k8ViISEhgZKSEgCio6PbrDcjoctut3PkyBGio6MJC+tZxFBA8cKg5GgiLGZq6rUmj4iIv6WlpQG4QoqcXMxmM5mZmT0OlgooXgizmBncL4bvi7Umj4iIv5lMJvr3709KSgr19fXBbo50UUREhE+mzFdAaan8EGxbCo31cMmv3XYNSenD98UV7NKaPCIiAWGxWHo8jkFOXhok21JtGXzye/h8CbQahXxGi3EoIiIi4l8KKC0lDgJMUFcO1cfcdmlNHhERkcBRQGkpPBLiTnM8Pv6j264hKe6VPCIiIuI/CiitJQ923LcKKK0reURERMR/FFBaS2oKKMd+cNvsrOQBXeYRERHxNwWU1pI896BA84RtmvJeRETEvxRQWusooDRNeb/rsHpQRERE/KnLAWXdunVMmjSJ9PR0TCYT77//vmtffX09jz76KCNGjCAmJob09HRuv/12Dh486HaOuro67r//fvr27UtMTAzXXHMN+/fv7/EP4xNJpzvuj//godTYEVD2lKgHRURExJ+6HFCqqqoYOXIkixcvbrOvurqarVu38sQTT7B1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YWFgGhUi1IMC2Gw2tmzZwoQJE9y2T5gwgY0bNwapVaGnrKwMgKSkpCC3JPimT5/OVVddxfjx44PdlJCwcuVKcnNzufHGG0lJSeG8887j1VdfDXazgurCCy/kH//4B7t27QLgq6++YsOGDVx55ZVBblloKCgooLi42O3vrtVqZezYsfq720pZWRkmk6nX9UielIsF+trRo0dpbGwkNTXVbXtqairFxcVBalVoMQyDmTNncuGFFzJ8+PBgNyeo3n77bbZu3cqmTZuC3ZSQ8eOPP7JkyRJmzpzJr3/9a7744gseeOABrFYrt99+e7CbFxSPPvooZWVlnHXWWVgsFhobG3n66af5xS9+EeymhQTn31ZPf3cLCwuD0aSQVFtby2OPPcaUKVNO6QUEPVFAacFkMrk9NwyjzbbeasaMGXz99dds2LAh2E0Jqn379vHggw+yevVqIiMjg92ckGG328nNzWXu3LkAnHfeeWzfvp0lS5b02oDyzjvvsHTpUpYtW8awYcPYtm0beXl5pKenc8cddwS7eSFDf3fbV19fzy233ILdbufFF18MdnMCTgEF6Nu3LxaLpU1vSUlJSZt03xvdf//9rFy5knXr1jFgwIBgNyeotmzZQklJCTk5Oa5tjY2NrFu3jsWLF1NXV4fFYgliC4Ojf//+DB061G3b2WefzXvvvRekFgXfrFmzeOyxx7jlllsAGDFiBIWFhcybN08BBUhLSwMcPSn9+/d3bdffXYf6+npuuukmCgoK+OSTT3pd7wmoigeAiIgIcnJyyM/Pd9uen5/P6NGjg9Sq4DMMgxkzZrB8+XI++eQTsrKygt2koLv00kv55ptv2LZtm+uWm5vL1KlT2bZtW68MJwBjxoxpU4K+a9cuBg4cGKQWBV91dTVms/ufWIvF0qvKjDuSlZVFWlqa299dm83G2rVre/XfXWgOJ7t372bNmjUkJycHu0lBoR6UJjNnzuS2224jNzeXf/u3f+OVV16hqKiIe+65J9hNC5rp06ezbNky/va3vxEbG+vqYYqPjycqKirIrQuO2NjYNmNwYmJiSE5O7tVjc371q18xevRo5s6dy0033cQXX3zBK6+8wiuvvBLspgXNpEmTePrpp8nMzGTYsGF8+eWXLFiwgDvvvDPYTQuYyspK9uzZ43peUFDAtm3bSEpKIjMzk7y8PObOnUt2djbZ2dnMnTuX6OhopkyZEsRW+19Hn0t6ejo33HADW7du5YMPPqCxsdH1tzcpKYmIiIhgNTvwgltEFFr+/Oc/GwMHDjQiIiKM888/v9eX0wIeb6+//nqwmxZSVGbs8Pe//90YPny4YbVajbPOOst45ZVXgt2koCovLzcefPBBIzMz04iMjDQGDx5sPP7440ZdXV2wmxYwn376qce/IXfccYdhGI5S49/97ndGWlqaYbVajYsvvtj45ptvgtvoAOjocykoKGj3b++nn34a7KYHlMkwDCOQgUhERESkMxqDIiIiIiFHAUVERERCjgKKiIiIhBwFFBEREQk5CigiIiISchRQREREJOQooIiIiEjIUUARERGRkKOAIiIiIiFHAUVERERCjgKKiIiIhBwFFBEREQk5/x9S8dYLK84v+AAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# loss\n",
    "plt.plot(history.history['loss'], label='train loss')\n",
    "plt.plot(history.history['val_loss'], label='val loss')\n",
    "plt.legend()\n",
    "plt.savefig('loss1.png')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "62a1332f",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('doc_datamodel.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "de63302d",
   "metadata": {},
   "outputs": [],
   "source": [
    "##############################################################\n",
    "import cv2\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.utils import load_img,img_to_array\n",
    "import numpy as np\n",
    "\n",
    "model=load_model('doc_datamodel.h5')\n",
    "img_width,img_height=224, 224"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e149a312",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "8bd2957a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Evaluate model on test data\n",
      "2/2 [==============================] - 2s 223ms/step - loss: 2.5427 - accuracy: 0.3913\n",
      "test loss, test acc: [2.5427112579345703, 0.3913043439388275]\n"
     ]
    }
   ],
   "source": [
    "print(\"Evaluate model on test data\")\n",
    "results = model.evaluate(test_x, test_y, batch_size=32)\n",
    "print(\"test loss, test acc:\", results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "6ad88355",
   "metadata": {},
   "outputs": [],
   "source": [
    "##prediction##s\n",
    "#pred_img.reshape(224,224)\n",
    "#x_pred=np.array([pred_img])\n",
    "#prediction=model.predict(x_pred)\n",
    "#print(prediction)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "16cc44b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#y_pred=np.argmax(x_pred,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "609da324",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test02.jpeg\n",
      "1/1 [==============================] - 0s 114ms/step\n",
      "1\n",
      "test01.jpeg\n",
      "1/1 [==============================] - 0s 99ms/step\n",
      "0\n",
      "test11.jpeg\n",
      "1/1 [==============================] - 0s 105ms/step\n",
      "1\n",
      "test20.jpeg\n",
      "1/1 [==============================] - 0s 69ms/step\n",
      "0\n",
      "test00.jpeg\n",
      "1/1 [==============================] - 0s 79ms/step\n",
      "3\n",
      "test21.jpeg\n",
      "1/1 [==============================] - 0s 71ms/step\n",
      "0\n",
      "test10.jpeg\n",
      "1/1 [==============================] - 0s 58ms/step\n",
      "1\n",
      "test22.jpeg\n",
      "1/1 [==============================] - 0s 51ms/step\n",
      "0\n",
      "test12.jpeg\n",
      "1/1 [==============================] - 0s 54ms/step\n",
      "0\n"
     ]
    }
   ],
   "source": [
    "pred_img=\"/home/fansan/Desktop/Document-Classifier/Dataset/testing/\"\n",
    "\n",
    "for img in os.listdir(pred_img):\n",
    "    img_path=pred_img+img\n",
    "    print(img)\n",
    "    img_arr=cv2.imread(img_path)\n",
    "    #print(pred_img.shape)\n",
    "    img_arr=cv2.resize(img_arr,(224, 224),3)\n",
    "    #print(pred_img.shape)\n",
    "    #print(type(pred_img))\n",
    "    img_arr=img_to_array(img_arr)\n",
    "    img_arr=np.expand_dims(img_arr,axis=0)\n",
    "    predictions=model.predict(img_arr)\n",
    "    p=predictions.argmax()\n",
    "    print(p)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "71e8871c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "None\n"
     ]
    }
   ],
   "source": [
    "print(pred_img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "104812e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/10.jpg (176, 287, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/11.jpg (268, 188, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/6.jpg (133, 289, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/3.jpg (624, 462, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/2.jpg (169, 298, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/0.jpg (179, 282, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/8.jpg (178, 269, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/1.jpg (176, 286, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/5.jpg (175, 287, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/9.jpg (188, 268, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/4.jpg (265, 190, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//passport/7.jpg (186, 271, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/10.jpg (255, 400, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/26.jpg (181, 279, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/23.jpg (323, 500, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/11.jpg (225, 225, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/22.jpg (352, 670, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/6.jpg (180, 281, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/25.jpg (800, 1200, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/20.jpg (337, 500, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/3.jpg (169, 250, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/2.jpg (184, 274, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/0.jpg (194, 326, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/13.jpg (178, 283, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/8.jpg (299, 500, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/1.jpg (177, 285, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/12.jpg (181, 279, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/15.jpg (180, 280, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/27.jpg (153, 243, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/16.jpg (189, 268, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/5.jpg (165, 257, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/19.jpg (176, 287, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/9.jpg (176, 287, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/24.jpg (181, 278, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/4.jpg (159, 248, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/7.jpg (175, 288, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/14.jpg (176, 225, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/21.jpg (179, 281, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//pan/17.jpg (178, 284, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/30.jpg (820, 1280, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/26.jpg (144, 153, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/40.jpg (505, 779, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/31.jpg (862, 1280, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/6.jpg (374, 561, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/41.jpg (220, 335, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/25.jpg (191, 319, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/3.jpg (1328, 885, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/2.jpg (503, 802, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/34.jpg (275, 428, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/0.jpg (734, 1152, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/13.jpg (176, 286, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/8.jpg (239, 358, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/33.jpg (161, 213, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/1.jpg (516, 774, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/12.jpg (749, 1200, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/37.jpg (696, 1086, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/15.jpg (496, 744, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/27.jpg (502, 746, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/5.jpg (465, 1000, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/39.jpg (370, 594, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/19.jpg (257, 347, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/32.jpg (334, 535, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/24.jpg (219, 365, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/29.jpg (486, 871, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/7.jpg (798, 1280, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/36.jpg (418, 624, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/18.jpg (397, 559, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/21.jpg (234, 338, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//driver license/17.jpg (457, 686, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/10.jpg (473, 710, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/28.jpg (178, 339, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/26.jpg (745, 1119, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/23.jpg (149, 224, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/11.jpg (2133, 3202, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/22.jpg (645, 965, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/6.jpg (433, 339, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/25.jpg (1427, 2091, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/20.jpg (320, 480, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/3.jpg (168, 300, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/2.jpg (360, 640, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/0.jpg (183, 275, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/13.jpg (297, 446, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/8.jpg (1937, 3025, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/1.jpg (259, 194, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/12.jpg (194, 259, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/15.jpg (129, 193, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/27.jpg (372, 559, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/16.jpg (184, 274, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/5.jpg (293, 231, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/19.jpg (77, 116, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/9.jpg (471, 705, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/24.jpg (124, 185, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/29.jpg (573, 860, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/4.jpg (194, 259, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/7.jpg (177, 237, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/18.jpg (739, 1059, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/14.jpg (176, 286, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/21.jpg (3120, 4160, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//aadhar/17.jpg (564, 865, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/40.png (475, 520, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/12.jpeg (1127, 1420, 3)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa4.jpg (2048, 1152, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/38.png (475, 441, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/27.png (282, 212, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/36.png (371, 265, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/10.jpg (300, 256, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa45.png (679, 513, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa20.jpeg (1011, 716, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/11.jpg (220, 172, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/23.png (254, 175, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/49.png (428, 545, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa28.png (425, 299, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/50.png (533, 533, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/46.png (923, 701, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/33.png (699, 423, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa30.png (439, 314, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa7.jpg (956, 611, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa19.jpeg (899, 624, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/34.png (374, 285, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/44.png (648, 438, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa8.jpg (282, 179, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa32.png (390, 279, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/29.png (453, 407, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa22.png (391, 260, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/13.jpg (2320, 1184, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/8.jpg (245, 209, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa31.png (332, 254, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa21.png (406, 282, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa42.png (308, 208, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/14.JPG (595, 526, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/35.png (331, 331, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/37.png (289, 289, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/3.jpeg (1280, 779, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/9.jpg (174, 130, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/39.png (413, 284, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa5.jpg (1280, 790, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/47.png (550, 344, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/16.PNG (863, 588, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/26.png (507, 440, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/18.jpg (960, 1280, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/43.png (696, 471, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa48.png (1125, 604, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/24.png (544, 407, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/aaa41.png (372, 274, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/train//voter/17.jpg (1160, 868, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//passport/3.jpg (633, 464, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//passport/2.jpg (453, 701, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//passport/0.jpg (622, 460, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//passport/1.jpg (130, 190, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/10.jpg (419, 695, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/6.jpg (300, 505, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/3.jpg (442, 673, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/2.jpg (358, 554, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/0.jpg (749, 1200, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/8.jpg (1219, 1219, 4)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/1.jpg (1302, 1301, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/5.jpg (311, 500, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/9.jpg (1168, 1169, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/4.jpg (768, 1280, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//dl/7.jpg (213, 346, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/6.jpg (259, 194, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/3.jpg (181, 279, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/2.jpg (555, 824, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/0.jpg (151, 335, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/8.jpg (187, 269, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/1.jpg (167, 301, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/5.jpg (183, 275, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/4.jpg (545, 970, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//aadhaar/7.jpg (81, 128, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/6.jpg (375, 600, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/3.jpg (206, 245, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/2.jpg (900, 1600, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/0.jpg (1123, 1500, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/8.jpg (412, 640, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/1.jpg (183, 275, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/5.jpg (194, 259, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/4.jpg (433, 770, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//pan/7.jpg (206, 245, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/aaa4.jpg (401, 157, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/10.jpg (254, 254, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/aaa3.jpg (752, 604, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/6.jpg (419, 419, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/aaa7.jpg (250, 160, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/0.jpg (1152, 648, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/13.jpg (216, 249, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/8.jpg (326, 325, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/1.jpg (443, 442, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/12.jpg (713, 451, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/9.jpg (470, 470, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/aaa5.jpg (503, 339, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/test//voter/aaa2.jpg (801, 485, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//passport/2.jpg (2000, 1500, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//passport/0.jpg (621, 455, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//passport/1.jpg (635, 460, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//dl/3.jpg (419, 695, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//dl/2.jpg (2650, 1844, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//dl/0.jpg (168, 300, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//dl/1.jpg (1963, 1684, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//dl/4.jpg (547, 835, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/6.jpg (168, 300, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/3.jpg (374, 500, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/2.jpg (185, 273, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/0.jpg (194, 259, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/1.jpg (168, 300, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/5.jpg (259, 194, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/4.jpg (547, 835, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//aadhaar/7.jpg (399, 574, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//pan/3.jpg (179, 281, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//pan/2.jpg (648, 1002, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//pan/0.jpg (152, 250, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//pan/1.jpg (405, 539, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//pan/5.jpg (181, 279, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//pan/4.jpg (178, 283, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//voter/10.jpg (1080, 732, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//voter/aaa3.jpg (1200, 829, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//voter/aaa9.jpg (450, 600, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//voter/2.jpg (595, 526, 3)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//voter/0.jpg (2320, 1184, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//voter/aaa22.png (391, 260, 3)\n",
      "/home/fansan/Desktop/Document-Classifier/Dataset/data/valid//voter/1.jpg (956, 660, 4)\n"
     ]
    }
   ],
   "source": [
    "x_train=[]\n",
    "\n",
    "for folder in os.listdir(train_path):\n",
    "\n",
    "    sub_path=train_path+\"/\"+folder\n",
    "\n",
    "    for img in os.listdir(sub_path):\n",
    "        img_path=sub_path+\"/\"+img\n",
    "        img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)\n",
    "        print(img_path,img.shape)\n",
    "\n",
    "x_test=[]\n",
    "\n",
    "for folder in os.listdir(test_path):\n",
    "    sub_path=test_path+\"/\"+folder\n",
    "    for img in os.listdir(sub_path):\n",
    "        img_path=sub_path+\"/\"+img\n",
    "        img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)\n",
    "        print(img_path,img.shape)\n",
    "\n",
    "x_val=[]\n",
    "\n",
    "for folder in os.listdir(val_path):\n",
    "\n",
    "    sub_path=val_path+\"/\"+folder\n",
    "\n",
    "    for img in os.listdir(sub_path):\n",
    "        img_path=sub_path+\"/\"+img\n",
    "        img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)\n",
    "        print(img_path,img.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c805b1b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
