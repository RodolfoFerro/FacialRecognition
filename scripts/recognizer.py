from tqdm import tqdm
import numpy as np
import glob
import cv2


# Path to Haar Cascades:
cvpath = "../haarcascades/"
casc = cv2.CascadeClassifier(cvpath + "haarcascade_frontalface_default.xml")

# Subjects:
subjects = ["Rodolfo Ferro"]

# Fisher Face classifier:
face_rec = cv2.face.createLBPHFaceRecognizer()


def recognize_person():
    # Create datasets:
    train_lbls, train_data = [], []
    train = sorted(glob.glob("../._db/cleaned/*"))
    for item in train:
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_data.append(gray)
        train_lbls.append(0)

    # Train model:
    print("\n\n{:=<40}".format(""))
    print("Train LBHP Face Classifier.")
    print("Size of training set is {} images.".format(len(train_lbls)))
    face_rec.train(train_data, np.array(train_lbls))

    # Load video capture:
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame:
        ret, img = cap.read()

        # Convert to grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face using 4 different classifiers:
        faces = casc.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=10,
                minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        for i, (x, y, w, h) in enumerate(faces):
            # Crop face:
            out = gray[y:y + h, x:x + w]

            # Resize & save face:
            out = cv2.resize(out, (350, 350))

            # Predict face:
            pred = face_rec.predict(out)
            cv2.rectangle(img, (x, y), (x + w, y + h), (180, 255, 10), 2)

            if pred[1] < 45:
                # cv2.putText(img, 'Face {}'.format(subjects[pred[0]]),
                cv2.putText(img, '{}'.format(subjects[0]),
                            (x + w//6, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.003*w, (255, 180, 10), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, 'Not recognized',
                            (x + w//6, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.003*w, (255, 180, 10), 2, cv2.LINE_AA)
            # print("Face: {}. Pred: {}".format(i, pred))

        # Display the resulting frame
        new = cv2.resize(img, (640, 360))
        cv2.imshow('Out', new)
        key = cv2.waitKey(10)
        if key == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_person()
