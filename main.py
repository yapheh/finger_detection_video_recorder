import cv2
import numpy as np

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def removeFaceArea(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray, cascade)

    for x1, y1, x2, y2 in rects:
        img = cv2.rectangle(img, (x1-10, y1-10), (x2+10, y2+10), (0,0,0), -1)

    return img

def make_mask_image(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = (0, 30, 0)
    high = (15, 255, 255)
    img_mask = cv2.inRange(img_hsv, low, high) # 색상 범위 내 이진 영상
    return img_mask

def findmaxArea(contours): # 추출된 윤곽선 중 가장 큰 윤곽선 검출
    max_contour = None
    max_area = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10000:
        max_area = -1

    return max_area, max_contour

def calculateAngle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product) * (180.0 / np.pi)
    return angle

def distanceBetweenTwoPoints(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def getFingerPosition(max_contour, img_result, draw):
    # 무게중심점 구하기
    M = cv2.moments(max_contour)
    if M['m00'] == 0:
        return -1, None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 윤곽선 근사 후 블록 껍질 검출
    max_contour = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    hull = cv2.convexHull(max_contour)

    # 윤곽선의 점이 무게중심보다 위에 있는지 판별
    points1 = []
    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))

    if draw:
        cv2.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
        for point in points1:
            cv2.circle(img_result, tuple(point), 15, [255, 0, 0], 3)

    hull = cv2.convexHull(max_contour, returnPoints=False)

    if len(hull) < 3:
        return -1, None

    try:
        defects = cv2.convexityDefects(max_contour, hull)
    except cv2.error as e:
        print(f"Convexity defects error: {e}")
        return -1, None

    if defects is None:
        return -1, None

    # 두 점간에 각도 판별
    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)
            if end[1] < cy:
                points2.append(end)

    points = points1 + points2
    points = list(set(points))

    final_points = []
    for point in points:
        prev = -1
        next = -1
        index = -1
        for i , c_point in enumerate(max_contour):
            cont_point = tuple(c_point[0])

            if point == cont_point:
                index = i
                break

        if index >= 0:
            if index == 0:
                prev = len(max_contour) - 1
            else:
                prev = index - 1

            if index == len(max_contour) - 1:
                next = 0
            else:
                next = index + 1

        prev_point = tuple(max_contour[prev][0])
        next_point = tuple(max_contour[next][0])

        angle = calculateAngle(np.array(prev_point) - np.array(point), np.array(next_point) - np.array(point))
        if angle < 90:
            final_points.append(point)

    if draw:
        cv2.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
        for point in points2:
            cv2.circle(img_result, tuple(point), 20, [0, 0, 255], 5)

    finger = len(final_points)
    cv2.putText(img_bgr, f"Finger : {finger}", (10, 50), 0, 1, (255, 0, 0), 1, cv2.LINE_AA)

    return 1, final_points


# main

# 얼굴 인식 학습 데이터 불러오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 웹캠에서 비디오 캡처
cap = cv2.VideoCapture(0) # (640, 480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

is_recording = False
out = None  # 초기화
finger_flag = False

# 컨투어, 포인트 그리기 여부
draw = True

while True:
    ret, img_bgr = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    # 스페이스바를 눌렀을 때
    if key == ord(' '):
        if not is_recording:
            # 비디오 저장 시작
            out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
            is_recording = True
            print("Recording started.")
        else:
            # 비디오 저장 종료
            out.release()
            is_recording = False
            print("Recording stopped.")

    # 비디오 저장 중이면 프레임 저장
    if is_recording:
        cv2.circle(img_bgr, (30, 30), 10, (0, 0, 255), -1)  # 빨간원 그리기
        cv2.putText(img_bgr, "RECORD", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(img_bgr)

    if key == ord('f'):
        finger_flag = not finger_flag

    if finger_flag == True:
        img_no_faces = removeFaceArea(img_bgr, face_cascade)
        img_mask = make_mask_image(img_no_faces)

        # 마스크 이미지에서 윤곽선 찾기
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_binary = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel) # 팽창 후 침식
        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area, max_contour = findmaxArea(contours)

        if max_area > 0:
            status, finger_positions = getFingerPosition(max_contour, img_bgr, draw)
            if status == 1:
                for pos in finger_positions:
                    cv2.circle(img_bgr, pos, 10, (0, 255, 0), -1)

    # 결과 화면 표시
    cv2.imshow('Finger Count recoder', img_bgr)

    # 'esc' 키를 누르면 종료
    if key == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
