import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import glob


# Container Definitions
header = st.container()
dataset = st.container()
modelVal = st.container()
model = st.container()
your_image = st.container()


# Load YOLO model and make predictions
def predict_model(model, img_in, conf):
    # Model paths
    model1_path = "res/yolov8n_EcoVision_full/runs/detect/train/weights/best.pt"
    model2_path = "res/yolov8n_EcoVision_with_augmentations/runs/detect/train/weights/best.pt"

    # Choose the model based on user selection
    if model == 'aug':
        model = YOLO(model2_path)
    elif model == 'org':
        model = YOLO(model1_path)

    # Perform prediction
    results3 = model.predict(source=img_in, conf=conf)
    img3 = results3[0].orig_img
    boxes3 = results3[0].boxes

    # Display the predicted bounding boxes and labels
    fig3, ax3 = plt.subplots()
    for box_ in boxes3:
        box2 = box_[0].data[0]
        tx3, ty3, bx3, by3 = int(box2.data[0]), int(box2.data[1]), int(box2.data[2]), int(box2.data[3])
        box_class3 = int(box2.data[5])
        prob3 = float(box2.data[4])
        box_cv3 = cv2.rectangle(img3, (tx3, ty3), (bx3, by3), colors[box_class3], 5)
        cv2.putText(box_cv3, f"{labels[box_class3]}, {prob3:.2f}", (int(tx3) + 10, int(ty3) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[box_class3], 1)
        ax3.set_axis_off()
        ax3.imshow(box_cv3)

    return fig3, len(boxes3), img3

with header:
    st.title('AcoVision')
    st.text("""
    EcoVision is an application designed to assist users in detecting and classifying 
    various types of garbage items, providing information on proper disposal methods
    based on the Israeli waste system. 
    The app utilizes the YOLOv8 nano detection and classification computer vision model
    for efficient identification of different types of garbage items.
    """)

with dataset:
    st.header('Dataset analysis')
    st.text("""
    EcoVision utilizes a vast collection of labeled images to retrain two versions 
    of the YOLOv8 nano models: the Original Model, without augmentation, 
    and the Augmented Model, with enhanced diversity through background replacement
    and noise addition. These modifications were made to enhance the model's performance 
    in real-world scenarios.
    """)

    mod_data = st.radio("Selecting a model for dataset analysis:", ('Original', 'Augmented'))
    st.subheader('Data and augmented data examples:')
    examples_paths = sorted(glob.glob('res/augmentations_examples/*'))
    image_indx0 = st.slider('Examples:', 0, 7, 0)

    if mod_data == 'Original':
        st.image(examples_paths[image_indx0+8])
        st.subheader('Instansces analysis')
        st.image('res/yolov8n_EcoVision_full/runs/detect/yolov8n_EcoVision_full7/labels.jpg')
    else:
        st.image(examples_paths[image_indx0])
        st.subheader('Instansces analysis')
        st.image('res/yolov8n_EcoVision_with_augmentations/runs/detect/yolov8n_EcoVision_fuller3 - best/labels.jpg')
with modelVal:
    st.header('Validation metrics')
    mod_val = st.radio("Selecting a model for the validation metrics overview:", ('Original', 'Augmented'))

    if mod_val == 'Original':
        # before augmentations (original)
        st.subheader('Original dataset')
        st.image('res/yolov8n_EcoVision_full/val2_conf_0.4_best/confusion_matrix.png')
        st.image("res/yolov8n_EcoVision_full/val2_conf_0.4_best/PR_curve.png")
        st.image("res/yolov8n_EcoVision_full/val2_conf_0.4_best/F1_curve.png")



    # after augmentations
    if mod_val == 'Augmented':
        st.subheader('Augmented dataset')
        st.image('res/yolov8n_EcoVision_with_augmentations/runs/detect/yolov8n_EcoVision_fuller3 - best/confusion_matrix.png')
        st.image("res/yolov8n_EcoVision_with_augmentations/runs/detect/yolov8n_EcoVision_fuller3 - best/PR_curve.png")
        st.image("res/yolov8n_EcoVision_with_augmentations/runs/detect/yolov8n_EcoVision_fuller3 - best/F1_curve.png")

with model:

    labels = {0: 'paper', 1: 'general', 2: 'packaging', 3: 'glass',
              4: 'cardboard', 5: 'organic', 6: 'deposit bottles', 7: 'electronics',
              8: 'clothing', 9: 'batteries', 10: 'medicines', 11: 'Light bulbs'}
    colors = {0: (0, 0, 255), 1: (209, 200, 200), 2:  (255, 128, 0), 3: (102, 0, 204),
          4: (153, 204, 255), 5: (0, 255, 0), 6: (100, 200, 100), 7: (0, 0, 0),
          8: (255, 51, 153), 9: (204, 204, 0), 10: (153, 153, 255), 11: (255, 255, 0)}

    # image to predict

    images_to_predict_list = glob.glob("images_to_predict/*")
    # print(images_to_predict_list[0])
    st.header('Prediction Examples')
    img_mod, colors_mod1, colors_mod2 = st.columns(3, gap="large")
    image_indx = img_mod.slider('Select an image for the example:', 0, len(images_to_predict_list)-1, 0)
    img1_path = images_to_predict_list[image_indx]
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img_mod.subheader('Original image')
    img_mod.image(img1)

    colors_mod1.subheader('Classes List:')
    count = 0
    for indx, label in enumerate(labels.values()):
        count += 1
        new_title = f'<p style="font-family:sans-serif; color:rgb{colors[indx]}; font-size: 18px;"> ------ {label}</p>'
        if count < 7:
            colors_mod1.markdown(new_title, unsafe_allow_html=True)
        else:
            colors_mod2.markdown(new_title, unsafe_allow_html=True)

    choose_mod_col, choose_con_col = st.columns(2)
    mod_ = choose_mod_col.radio("Selecting a model for prediction:", ('Original', 'Augmented'))
    confidence = choose_con_col.slider('Setting the minimum confidence level:', 0.0, 1.0, 0.35, 0.05)

    if mod_ == 'Original':
        fig, len1, img = predict_model('org', img1, confidence)
        st.subheader('Original model results')
        if len1 == 0:
            st.text('No detections')
            st.image(img)
        else:
            st.text(f'{str(len1)} detections')
            st.pyplot(fig)

    if mod_ == 'Augmented':
        fig2, len2, img2 = predict_model('aug', img1, confidence)
        st.subheader('Augmented model results')
        if len2 == 0:
            st.text('No detections')
            st.image(img2)
        else:
            st.text(f'{str(len2)} detections')
            st.pyplot(fig2)

with your_image:
    st.title("It's your turn!")
    uploaded_file = st.file_uploader("Upload an image (png\jpg\jpeg\webp):", type=['jpeg', 'png', 'jpg', 'webp'])
    mod_your_image = st.radio("Choose your poison:", ('Original', 'Augmented'))
    confidence2 = st.slider('How confident are you?', 0.0, 1.0, 0.35, 0.05)
    if uploaded_file == None:
        st.warning('No file selected')
    else:
        if mod_your_image=='Original':
            model='org'
            st.subheader('Augmented model')
        else:
            model = 'aug'
            st.subheader('Augmented model')
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig3, len3, img3 = predict_model(model, image, confidence2)

        if len3 == 0:
            st.text('No detections')
            st.image(img3)
        else:
            st.text(f'{str(len3)} detections')
            st.pyplot(fig3)
