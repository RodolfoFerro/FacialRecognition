from tqdm import tqdm
import glob
import cv2


# Path to Haar Cascades:
cvpath = "../haarcascades/"
face_det_01 = cv2.CascadeClassifier(
    cvpath + "haarcascade_frontalface_default.xml")
face_det_02 = cv2.CascadeClassifier(
    cvpath + "haarcascade_frontalface_alt2.xml")
face_det_03 = cv2.CascadeClassifier(
    cvpath + "haarcascade_frontalface_alt.xml")
face_det_04 = cv2.CascadeClassifier(
    cvpath + "haarcascade_frontalface_alt_tree.xml")

label = "Rod"


def extract_faces():
    files = sorted(glob.glob("../._db/original/*"))

    filenumber = 1
    for f in tqdm(files):
        img = cv2.imread(f)

        # Convert to grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face using 4 different classifiers:
        face_01 = face_det_01.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_02 = face_det_02.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_03 = face_det_03.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_04 = face_det_04.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        # Detect only one face:
        if len(face_01) == 1:
            facefeatures = face_01
        elif len(face_02) == 1:
            facefeatures = face_02
        elif len(face_03) == 1:
            facefeatures = face_03
        elif len(face_04) == 1:
            facefeatures = face_04
        else:
            facefeatures = ""

        for (x, y, w, h) in facefeatures:
            # pri3t("Face found in file: {}".format(f))
            # Crop face:
            new = gray[y:y + h, x:x + w]

            try:
                # Resize & save face:
                out = cv2.resize(new, (350, 350))
                cv2.imwrite("../._db/cleaned/{}.jpg".format(filenumber), out)
            except:
                pass
        filenumber += 1


if __name__ == "__main__":
    extract_faces()
