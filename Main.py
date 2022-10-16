import cv2
import mediapipe as mp
import copy
import Functions as fn
import HandPoint as hp
import numpy as np
import pyshine as ps

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

LbPath = 'D:\KLTN\1_main\model\Hand_Point\HandPose.csv'
pose = fn.ReadLb(LbPath)
hpc = hp.HPC()

cap = cv2.VideoCapture(0)
j = 0
ip = -1
text  =  'Input: '
while True:
    success,img=cap.read()
    img = cv2.flip(img, 1) # Đảo ảnh
    # Xử lý ảnh
    # kernel = np.array([[0, -1, 0],
    #                [-1, 5,-1],
    #                [0, -1, 0]])
    # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    # img = cv2.GaussianBlur(img, (5,5), 2,2)

    List = []

    debug_img = copy.deepcopy(img)
    imgRGB = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks
        # for id, lm in enumerate(myHand):
        #     mpDraw.draw_landmarks(debug_img, lm, mpHands.HAND_CONNECTIONS, 
        #                             mpDraw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #                             mpDraw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
        #                             )
                                    
        lml = fn.FindPoint(debug_img, myHand)
        for lm, handedness in zip(myHand, results.multi_handedness):
            brect = fn.CalcBR(debug_img, lm)

        

        pre = fn.pre_process(lml)

        hand_sign_id, test = hpc(pre)

        debug_img = fn.DrawBR(True, debug_img, brect)
        if test[hand_sign_id] >= 0.85:
            debug_img = fn.DrawIT(
                        debug_img,
                        brect,
                        handedness,
                        pose[hand_sign_id]
                    )
            if ip != hand_sign_id:
                ip = hand_sign_id
                j = 0
            else:
                if j == 10:
                    text  =  text + pose[hand_sign_id]
                    j = 0
                else: 
                    j += 1
        else:
            debug_img = fn.DrawIT(
                        debug_img,
                        brect,
                        handedness,
                        ""
                    )

    debug_img = ps.putBText(debug_img, text, text_offset_x=20, text_offset_y=20, vspace=10, hspace=10, font_scale=1.0, background_RGB=(228,225,222), text_RGB=(1,1,1))
    cv2.imshow("Image",debug_img)

    key = cv2.waitKey(10)
    if key == 27:
        break

                        
cap.release()
cv2.destroyAllWindows()