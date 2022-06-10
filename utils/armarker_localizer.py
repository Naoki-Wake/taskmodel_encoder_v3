import numpy as np
import cv2
import cv2.aruco as aruco
import os
import json


def estimate_homo_transform_matrix(img_src):
    try:
        img = img_src.copy()
        fp_cameraparam = os.path.join(
            os.getcwd(), 'settings', 'marker_setting.json')
        # open json
        with open(fp_cameraparam) as f:
            data = json.load(f)
        _camera_matrix = np.array(data['camera_K']).reshape([3, 3])
        _dist_coef = np.array(data['camera_D'])
        # key = getattr(aruco, 'DICT_6X6_250')
        key = getattr(aruco, data['marker_type'])
        marker_size = data['marker_size']
        marker_id = data['marker_id']
        # gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Image', gray)
        # cv2.waitKey(0)
        dict = aruco.Dictionary_get(key)
        param = aruco.DetectorParameters_create()
        bboxs, ids, rejected = aruco.detectMarkers(
            gray, dict, parameters=param)
        aruco.drawDetectedMarkers(img, bboxs)
        if len(bboxs) != 0:
            for bbox, id in zip(bboxs, ids):
                print('Detected marker id: ' + str(int(id)))
                if int(id) == marker_id:
                    # pose estimation
                    rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(
                        bbox, marker_size, _camera_matrix, _dist_coef)
                    aruco.drawAxis(
                        img, _camera_matrix, _dist_coef, rvecs, tvecs, 0.05)
                    R, _ = cv2.Rodrigues(rvecs[0])
                    T = np.array(tvecs[0][0])
                    rot_mat_4x4 = np.zeros((4, 4))
                    rot_mat_4x4[:3, :3] = R
                    rot_mat_4x4[:3, 3] = T
                    rot_mat_4x4[3, 3] = 1
                    # inverse
                    rot_mat_4x4_marker_to_camera = np.linalg.inv(rot_mat_4x4)
                    # print(rot_mat_4x4_marker_to_camera)
                    return rot_mat_4x4_marker_to_camera, img

        return None, img
    except Exception as err:
        print(err)
        return None, None


if __name__ == '__main__':
    # fp_data = "D://Dataset_stop_and_go//tmp//ar marker//task_recognition//ar marker.mp4"
    fp_data = "D://Dataset_stop_and_go//withaudio//pick_place_07//task_recognition//segment_part_0-62.mp4"
    # read frame from video
    cap = cv2.VideoCapture(fp_data)
    ret, frame = cap.read()
    rot_mat_4x4_marker_to_camera, frame = estimate_homo_transform_matrix(frame)
    cv2.imshow('Image', frame)
    cv2.waitKey(0)
