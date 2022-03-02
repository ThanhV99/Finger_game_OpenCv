from cvzone.HandTrackingModule import HandDetector
import cv2
import time
import cvzone
import random
import numpy as np

def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype=np.uint8)   # cast pixel values to int
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    return img_new

# do mau
def FocalLength(pixel_length = 180, meansured_distance = 19, real_length = 6.5):
    focal_length = int((pixel_length*meansured_distance)/real_length)
    return focal_length

def Caculate_real(pixel_legth, real_length = 6.5):
    focal_legth = FocalLength()
    real_distance = int((focal_legth/pixel_legth)*real_length)
    return real_distance

class Enemy:
    def __init__(self):
        self.x = random.randint(40, w_screen-40)
        self.y = random.randint(40, h_screen-150)
        self.radius = 20
        self.color = PURPLE_COLOR

    def creat_enemy(self):
        self.x = random.randint(40, 600)
        self.y = random.randint(40, 440)
        self.color = PURPLE_COLOR

    def draw(self):
        cv2.circle(final_img, (self.x, self.y), self.radius, self.color, -1)
        cv2.circle(final_img, (self.x, self.y), self.radius-10, WHITE_COLOR, 3)


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
pTime = 0
w_screen = 800
h_screen = 600

PURPLE_COLOR = (255,0,255)
WHITE_COLOR = (255,255,255)
GREEN_COLOR = (0,255,0)

#bien tro choi
ls_enemy = []
enemy = Enemy()
score = 0
push = False
check = False
distance_detect = 25
range_finger = 40

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.resize(img, dsize=(w_screen, h_screen))
    img = cv2.flip(img, 1)
    img_new = change_brightness(img, 0.3, 5)
    final_img = img_new

    # Find the hand and its landmarks
    # hands, img = detector.findHands(img)  # with draw
    hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList = hand1["lmList"]  # List of 21 Landmark points
        bbox = hand1["bbox"]  # Bounding box info x,y,w,h
        center_bbox = hand1['center']
        # handType1 = hand1["type"]  # Handtype Left or Right
        pixel_legth, info = detector.findDistance(lmList[5][:2], lmList[17][:2])
        x, y = lmList[8][0], lmList[8][1]

        roi = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        roi = cv2.circle(roi, (x, y), range_finger, 255, -1)

        mask_roi = cv2.bitwise_and(img, img, mask=roi)
        bg_roi = cv2.bitwise_and(img_new, img_new, mask=~roi)

        final_img = cv2.add(mask_roi, bg_roi)

        real_distance = Caculate_real(pixel_legth)

        if real_distance < distance_detect and push == False:
            push = True
            if x - range_finger < enemy.x < x + range_finger and y - range_finger < enemy.y < y + range_finger:
                enemy.color = GREEN_COLOR

        if real_distance > distance_detect:
            push = False

        if enemy.color == (0,255,0) and push == False:
            enemy.creat_enemy()
            score += 1

        cvzone.putTextRect(final_img, f'{real_distance} cm', (x+5, y-15), scale=2, thickness=1)

    enemy.draw()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cvzone.putTextRect(final_img, f'Score: {score}', (10, 30), scale=2, thickness=2)
    cv2.putText(final_img, f'FPS: {int(fps)}', (20, h_screen-20), cv2.FONT_HERSHEY_COMPLEX,
                0.7, (255, 0, 0), 1)

    cv2.imshow('Game', final_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()