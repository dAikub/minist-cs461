import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import streamlit as st


st.markdown("""
    <style>
        .stButton>button {
            background-color: #FA6969;  /* เปลี่ยนสีปุ่ม */
            color: white;               /* เปลี่ยนสีข้อความในปุ่ม */
            font-size: 16px;            /* ขนาดฟอนต์ */
            padding: 10px 20px;         /* ขนาดปุ่ม */
            border-radius: 20px;         /* มุมโค้ง */
            width: 100%;                /* กำหนดความกว้างของปุ่ม */
            cursor: pointer;           /* เปลี่ยนลักษณะเมาส์เป็น pointer เมื่อเอาเมาส์ไปวาง */
            transition: all 0.3s ease;  /* เพิ่ม transition ให้การเปลี่ยนแปลง */
        }
        .stButton>button:hover {
            background-color: #C3E8A6;  /* สีเมื่อเอาเมาส์ไปวาง */
            color: white;              /* เปลี่ยนสีฟอนต์เมื่อเอาเมาส์ไปวาง */
            border: 2px solid #C3E8A6;  /* ขอบสีทองเมื่อเอาเมาส์ไปวาง */
        }
        .stButton>button:active {
            background-color: #FF1493;  /* สีพื้นหลังเมื่อกดปุ่ม */
            color: white;               /* สีข้อความเมื่อกดปุ่ม */
            border: 2px solid #8B0000;  /* ขอบสีแดงเข้มเมื่อกดปุ่ม */
        }
        /* จัดตำแหน่งปุ่มตรงกลาง */
        .stButton {
            display: flex;
            justify-content: center;
    
        }
    </style>
    """, unsafe_allow_html=True)


st.header("วิเคราะห์รูปภาพประเภทของเครื่องแต่งกาย")
st.write(' ')
st.write(' ')
st.write(' ')



fashion_dict = {
    0: "เสื้อยืด",
    1: "กางเกง",
    2: "เสื้อฮูด",
    3: "เดรส",
    4: "เสื้อคลุม",
    5: "รองเท้าแตะ",
    6: "เสื้อเชิ้ต",
    7: "รองเท้าผ้าใบ",
    8: "กระเป๋า",
    9: "รองเท้าบูทหุ้มข้อ"
}

# ตัวอย่างการใช้งาน
print(fashion_dict[3])  # Output: Dress



(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

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
    data = st.slider("เลือกรูปที่",1,50,label_visibility='hidden')
    plt.imshow(x_train[data-1],cmap = plt.cm.binary)
    plt.axis('off')
    st.pyplot(plt.gcf())

    prediction = model.predict(x_train)
    result  = np.argmax(prediction[data-1])
    matched_results = [fashion_dict[result]]

    enter = st.button("วิเคราะห์")
    if enter:
        st.success(f"ภาพนี้คือภาพ {matched_results[0]}")




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
        
        # จัดข้อความให้อยู่ตรงกลาง
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 16px; margin-top: 10px; margin-bottom: 10px;">
                รูปที่ {i+1}
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.write('---')
