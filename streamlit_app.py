import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import streamlit as st


st.header("วิเคราะห์รูปภาพตัวเลขข้อมูล Minist")
st.write(' ')
st.write(' ')
st.write(' ')

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255

if 'model_trained' not in st.session_state:



    model = Sequential()
    model.add(Flatten(input_shape=[28,28]))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1)
    acc = model.evaluate(x_test, y_test)

    st.session_state.model_trained = True
    st.session_state.model = model
else:
    model = st.session_state.model


    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(x_train, y_train, epochs=1)
   # acc = model.evaluate(x_test, y_test)

    
with st.sidebar:
    st.header("เลือกรูป")
    data = st.slider("data",1,50,label_visibility='hidden')
    plt.imshow(x_train[data-1],cmap = plt.cm.binary)
    plt.axis('off')
    st.pyplot(plt.gcf())
    enter = st.button("วิเคราะห์")
    if enter:
        prediction = model.predict(x_train)
        st.success(f"ตัวเลขคือเลข {np.argmax(prediction[data-1])}")




cols = st.columns(4)

plt.figure(figsize=(3, 3))  # Adjust the figure size as needed

if 'images_displayed' not in st.session_state:
    # กำหนดคอลัมน์ 6 คอลัมน์
    cols = st.columns(4)
    
    # แสดงภาพทั้งหมด
    for i in range(50):
        col_idx = i % 4
        with cols[col_idx]:
            # แสดงภาพ
            plt.imshow(x_train[i], cmap=plt.cm.binary)
            plt.axis('off')
            st.pyplot(plt.gcf())
            st.info(f"รูปที่ {i+1}")
            st.write('---')

