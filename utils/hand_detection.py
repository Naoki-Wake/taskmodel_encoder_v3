from ftplib import all_errors
import json
from time import time
from unicodedata import name
import cv2
import requests
import os
import tempfile
import numpy as np

# setting for mediapipe
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Mapping, Optional, Tuple, Union
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tomlkit import string
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
def _normalize_color(color):
    return tuple(v / 255. for v in color)

def plot_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: mp.solutions.drawing_utils.DrawingSpec = mp.solutions.drawing_utils.DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: mp.solutions.drawing_utils.DrawingSpec = mp.solutions.drawing_utils.DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10,
                   vizualize: bool = True,
                   output_path = None) -> None:
    """
    borrowed from https://github.com/google/mediapipe/blob/4a20e9909d55838d5630366ce719844cf06ae85c/mediapipe/python/solutions/drawing_utils.py#L257
    """
    if not landmark_list:
        return
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
            landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    # offset all points in plotted_landmarks
    triangle_x = np.array([plotted_landmarks[0][0], plotted_landmarks[5][0], plotted_landmarks[17][0]])
    triangle_y = np.array([plotted_landmarks[0][1], plotted_landmarks[5][1], plotted_landmarks[17][1]])
    triangle_z = np.array([plotted_landmarks[0][2], plotted_landmarks[5][2], plotted_landmarks[17][2]])
    offset_x = np.mean(triangle_x)
    offset_y = np.mean(triangle_y)
    offset_z = np.mean(triangle_z)
    keys = plotted_landmarks.keys()
    for i in keys:
        plotted_landmarks[i] = (plotted_landmarks[i][0] - offset_x, plotted_landmarks[i][1] - offset_y, plotted_landmarks[i][2] - offset_z)
    
    # draw points
    keys = plotted_landmarks.keys()
    for i in keys:
        xyz = plotted_landmarks[i]
        ax.scatter3D(
            xs=xyz[0],
            ys=xyz[1],
            zs=xyz[2],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness)  
        if idx in [0,5,17]:
            ax.scatter3D(
                xs=xyz[0],
                ys=xyz[1],
                zs=xyz[2],
                color=_normalize_color(RED_COLOR[::-1]),
                linewidth=landmark_drawing_spec.thickness)              
    # draw connections
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    triangle_x = np.array([plotted_landmarks[0][0], plotted_landmarks[5][0], plotted_landmarks[17][0]])
    triangle_y = np.array([plotted_landmarks[0][1], plotted_landmarks[5][1], plotted_landmarks[17][1]])
    triangle_z = np.array([plotted_landmarks[0][2], plotted_landmarks[5][2], plotted_landmarks[17][2]])
    # 1. create vertices from points
    verts = [list(zip(triangle_x, triangle_y, triangle_z))]
    # 2. create 3d polygons and specify parameters
    srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
    # 3. add polygon to the figure (current axes)
    plt.gca().add_collection3d(srf)

    v1 = np.array([triangle_x[1], triangle_y[1], triangle_z[1]]) - np.array([triangle_x[0], triangle_y[0], triangle_z[0]])
    v2 = np.array([triangle_x[2], triangle_y[2], triangle_z[2]]) - np.array([triangle_x[0], triangle_y[0], triangle_z[0]])
    # outer product of vectors
    v3 = np.cross(v1, v2)
    # normalize
    v3 = v3 / np.linalg.norm(v3)

    # define a vec
    vec_hand_z = np.array([np.mean(triangle_x),np.mean(triangle_y),np.mean(triangle_z)]) - np.array([triangle_x[0], triangle_y[0], triangle_z[0]])
    vec_hand_z = vec_hand_z / np.linalg.norm(vec_hand_z)
    vec_hand_y = -v3
    vec_hand_x = np.cross(vec_hand_y, vec_hand_z)
    rotation_matrix_hand = np.array([vec_hand_x, vec_hand_y, vec_hand_z])
    rotation_matrix_hand = rotation_matrix_hand.T
    # draw normal from center of triangle
    # plt.quiver(np.mean(triangle_x),np.mean(triangle_y),np.mean(triangle_z), v3[0], v3[1], v3[2], length=0.1, color=['#808080'])
    
    #draw normal from center of hand
    plt.quiver(np.mean(triangle_x),np.mean(triangle_y),np.mean(triangle_z), vec_hand_x[0], vec_hand_x[1], vec_hand_x[2], length=0.1, color=['#800000'])
    plt.quiver(np.mean(triangle_x),np.mean(triangle_y),np.mean(triangle_z), vec_hand_y[0], vec_hand_y[1], vec_hand_y[2], length=0.1, color=['#008000'])
    plt.quiver(np.mean(triangle_x),np.mean(triangle_y),np.mean(triangle_z), vec_hand_z[0], vec_hand_z[1], vec_hand_z[2], length=0.1, color=['#000080'])
    # add axis label
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    # same scale for all axes
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    if vizualize:
        plt.show()
    else:
        if output_path is not None:
            plt.savefig(output_path)
        plt.close()
    return rotation_matrix_hand, vec_hand_x, vec_hand_y, vec_hand_z

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(color=[0,255,0],thickness=1, circle_radius=1)
hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5)
def landmark_finder(fp_img,dir_detection_result):
    #mp_drawing = mp.solutions.drawing_utils
    #mp_hands = mp.solutions.hands
    #drawing_spec = mp_drawing.DrawingSpec(color=[0,255,0],thickness=1, circle_radius=1)
    #with mp_hands.Hands(
    #        static_image_mode=True,
    #        max_num_hands=1,
    #        min_detection_confidence=0.5) as hands:
    image = cv2.imread(fp_img)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)
    annotated_image = image.copy()
    if not results.multi_hand_landmarks:
        return None, None
    scores = [x.classification[0].score for x in results.multi_handedness]
    # argmax returns the index of the max value in the list
    max_score_index = scores.index(max(scores))
    hand_landmarks = results.multi_hand_landmarks[max_score_index]
    # print('hand_landmarks:', hand_landmarks)
    mp_drawing.draw_landmarks(
        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
    # show the image
    # cv2.imshow('Hand Detection', annotated_image)
    #mp_drawing.plot_landmarks(
    rotmat, vec_hand_x, vec_hand_y, vec_hand_z = plot_landmarks(    
        hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec,
        vizualize=False,
        output_path=os.path.join(dir_detection_result,os.path.basename(fp_img)))
    output_path=os.path.join(dir_detection_result,os.path.basename(fp_img).split('.')[0]+'_overlay.jpg')
    cv2.imwrite(output_path, annotated_image)
    # cv2.waitKey(0)
    #print('norm_vec:', rotmat)
    from pyquaternion import Quaternion
    q = Quaternion(matrix=rotmat)
    #print('quaternion:', q)
    #print('hand_x')
    #print(vec_hand_x)
    #print(q.rotate([1.0, 0.0, 0.0]))
    #print('hand_y')
    #print(vec_hand_y)
    #print(q.rotate([0.0, 1.0, 0.0]))
    #print('hand_z')
    #print(vec_hand_z)
    #print(q.rotate([0.0, 0.0, 1.0]))
    return q, max(scores)


def upload_data(mp4path, jsonpath):
    url = 'http://20.109.51.15:8083/hand_localization'
    headers = {'accept': 'application/json'}
    data = {
        'upload_file': open(
            mp4path, 'rb'), 'upload_json': open(
            jsonpath, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    #data = response.data()
    return response


def upload_data_image(fp_tmp_img):
    url = 'http://20.109.51.15:8083/hand_localization_image'
    headers = {'accept': 'application/json'}
    data = {'upload_file': open(fp_tmp_img, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    #data = response.data()
    return response


def run(time_focus, mp4path, output_dir):
    import os
    fp_tmp_json = os.path.join(output_dir, 'tmp.json')
    fp_tmp_out = os.path.join(output_dir, 'tmp.zip')
    json_send = {}
    json_send['time_focus'] = time_focus
    # save the result
    with open(fp_tmp_json, 'w') as outfile:
        json.dump(json_send, outfile, indent=4)
    # ask server to convert image to depth
    response = upload_data(mp4path, fp_tmp_json)
    data = response.content
    # save the result
    with open(fp_tmp_out, 'wb') as s:
        s.write(data)

    print(fp_tmp_out)
    import shutil
    try:
        shutil.unpack_archive(
            fp_tmp_out, os.path.join(
                output_dir, 'hand_detection'))
        from glob import glob
        import zipfile
        zip_f = zipfile.ZipFile(fp_tmp_out, 'r')
        lst = zip_f.namelist()
        relative_path = ""
        for item in lst:
            if "hand_detection.json" in item:
                relative_path = item
        return os.path.normpath(
            os.path.join(
                output_dir,
                'hand_detection',
                relative_path))
    except BaseException:
        print('unpack failed (hand_detection.py)')
    # find the result file


def run_allframe(mp4path, json_send, output_dir):
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        import os
        fp_tmp_json = os.path.join(output_dir_tmp, 'tmp.json')
        fp_tmp_out = os.path.join(output_dir, 'tmp.zip')
        # save the result
        with open(fp_tmp_json, 'w') as outfile:
            json.dump(json_send, outfile, indent=4)
        # ask server to convert image to depth
        response = upload_data(mp4path, fp_tmp_json)
        data = response.content
        # save the result
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)
        print(fp_tmp_out)
        import shutil
        try:
            shutil.unpack_archive(
                fp_tmp_out, os.path.join(
                    output_dir, 'hand_allframe'))            
            import zipfile
            zip_f = zipfile.ZipFile(fp_tmp_out, 'r')
            lst = zip_f.namelist()
            relative_path = ""
            for item in lst:
                if "hand_detection.json" in item:
                    relative_path = item
            fp_json = os.path.join(
                    output_dir,
                    'hand_allframe',
                    relative_path)
            with open(fp_json) as json_file:
                data = json.load(json_file)
                return data
        except BaseException:
            print('unpack failed (hand_detection.py)')
        # find the result file

def run_image(frame_img):
    with tempfile.TemporaryDirectory() as output_dir:
        # print(output_dir+ "in run_iamge")
        fp_tmp_img = os.path.join(output_dir, 'tmp.png')
        cv2.imwrite(fp_tmp_img, frame_img)

        fp_tmp_out = os.path.join(output_dir, 'tmp.json')
        response = upload_data_image(fp_tmp_img)
        data = response.content
        if data == b'Internal Server Error':
            # quit the program
            raise Exception('Internal Server Error')

        with open(fp_tmp_out, 'wb') as s:
            s.write(data)
        with open(fp_tmp_out) as json_file:
            data = json.load(json_file)
            return data

def run_allframe_with_graspdetection(mp4path, json_send, output_dir):
    def upload_data_with_grasp(mp4path, jsonpath):
        url = 'http://20.109.51.15:8083/hand_localization_with_grasp'
        headers = {'accept': 'application/json'}
        data = {
            'upload_file': open(
                mp4path, 'rb'), 'upload_json': open(
                jsonpath, 'rb')}
        response = requests.post(url, headers=headers,
                                files=data)
        #data = response.data()
        return response

    with tempfile.TemporaryDirectory() as output_dir_tmp:
        import os
        fp_tmp_json = os.path.join(output_dir_tmp, 'tmp.json')
        fp_tmp_out = os.path.join(output_dir, 'tmp.zip')
        # save the result
        with open(fp_tmp_json, 'w') as outfile:
            json.dump(json_send, outfile, indent=4)
        # ask server to convert image to depth
        response = upload_data_with_grasp(mp4path, fp_tmp_json)
        data = response.content
        # save the result
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)
        print(fp_tmp_out)
        import shutil
        try:
            shutil.unpack_archive(
                fp_tmp_out, os.path.join(
                    output_dir, 'hand_allframe'))            
            import zipfile
            zip_f = zipfile.ZipFile(fp_tmp_out, 'r')
            lst = zip_f.namelist()
            relative_path = ""
            for item in lst:
                if "hand_detection.json" in item:
                    relative_path = item
            fp_json = os.path.join(
                    output_dir,
                    'hand_allframe',
                    relative_path)
            with open(fp_json) as json_file:
                data = json.load(json_file)
                return data
        except BaseException:
            print('unpack failed (hand_detection.py)')
        # find the result file

if __name__ == '__main__':
    import numpy as np
    mode_hand_detection = False
    skip_frame=3
    #fp_mp4 = '../tmp/kitchen06_rescale.mp4'
    #fp_mp4 = '../tmp/trash04_rescale.mp4'
    #fp_mp4 = 'D:\\Dataset_stop_and_go\\Building99\\lab\\20225031\\drawer03\\task_recognition\\drawer03_rescale.mp4'
    #fp_mp4 = 'D:\\Dataset_nonstop\\Building99\\20220601\\pick_place01_v2\\task_recognition\\pick_place01.mp4'
    #fp_mp4 = '..\\tmp\\hand.mp4'
    fp_mp4 = '..\\tmp\\hand_rotation.mp4'
    print(os.path.isfile(fp_mp4))
    output_dir = '../output_user_interface/'
    fp_mp4_out = os.path.join(output_dir, 'tmp.mp4')
    cap = cv2.VideoCapture(str(fp_mp4))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #import pdb; pdb.set_trace()
    ret, _ = cap.read()
    #import pdb; pdb.set_trace()
    if ret:
        # video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_focus = np.arange(
        0,
        (all_frame-1) / float(fps),
        skip_frame / float(fps)).tolist()
    frame_focus = np.arange(
        0,
        all_frame,
        skip_frame).tolist()
    json_send = {}
    json_send['time_focus'] = time_focus
    if mode_hand_detection:
        writer = cv2.VideoWriter(fp_mp4_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        json_data = run_allframe_with_graspdetection(
            fp_mp4, json_send, output_dir)
        grasp_prob_right  = np.zeros(len(json_data))
        for i in range(len(json_data)):
            if len(json_data[i]['location_righthand']) > 0:
                if json_data[i]['predict_righthand'][0] == 'Grasp':
                    grasp_prob_right[i] = float(json_data[i]['predict_righthand'][1])
                elif json_data[i]['predict_righthand'][0] == 'Release':
                    grasp_prob_right[i] = 1-float(json_data[i]['predict_righthand'][1])
        # import pdb; pdb.set_trace()
        import matplotlib.pyplot as plt
        plt.plot(time_focus, grasp_prob_right)
        # moving average using pandas
        window_size = 5
        # import pdb; pdb.set_trace()
        import pandas as pd
        df = pd.DataFrame(grasp_prob_right)
        grasp_prob_right_ma = df.rolling(window=window_size, min_periods=1).mean()
        plt.plot(time_focus, grasp_prob_right_ma.values)
        grasp_prob_right_ma = grasp_prob_right_ma.values
        #import numpy as np
        #window_size = 5
        #weight = np.repeat(1.0, window_size) / window_size
        #grasp_prob_right_ma = np.convolve(grasp_prob_right, weight, 'valid')
        
        # where grasp_prob_right_ma iis change below 0.2 to above 0.2
        # import pdb; pdb.set_trace()
        sel_grasp = np.where((grasp_prob_right_ma[1:]>0.2) & (grasp_prob_right_ma[:-1]<=0.2))[0]
        sel_release = np.where((grasp_prob_right_ma[1:]<=0.2) & (grasp_prob_right_ma[:-1]>0.2))[0]
        print("grasp")
        for i in sel_grasp:
            print(time_focus[i])
        print("release")
        for i in sel_release:
            print(time_focus[i])

        #plt.plot(time_focus, grasp_prob_right_ma)
        #plt.show()
        cap.release()
        cap = cv2.VideoCapture(str(fp_mp4))
        ret, frame = cap.read()
        i = 0
        while ret:
            #import pdb; pdb.set_trace()
            current_time = i/float(fps)
            closest_time = np.argmin(np.abs(np.array(time_focus) - current_time))
            if grasp_prob_right_ma[closest_time]>0.2:
                cv2.putText(frame, "Grasp", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (30, 30, 255), 4)
            else:
                cv2.putText(frame, "Release", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (30, 30, 255), 4)
            writer.write(frame)
            ret, frame = cap.read()
            i += 1
        cap.release()
        writer.release()
    plt.show()
    # estimate hand keypoints
    #dir_detection_result = os.path.join(output_dir, 'hand_allframe','pick_place01')
    dir_detection_result = os.path.join(output_dir, 'hand_allframe','hand_rotation')
    fp_detected_hands = os.path.join(dir_detection_result,'hand_detection.json')
    dir_detection_result_save = os.path.join(dir_detection_result, 'out')
    if not os.path.exists(dir_detection_result_save):
        os.makedirs(dir_detection_result_save)
    json_data = json.load(open(fp_detected_hands))
    print(len(json_data))
    q_list = len(json_data) * [None]
    scores = len(json_data) * [0]
    time_frame = np.array(len(json_data) * [0.0])
    for i, item in enumerate(json_data):
        time_frame[i] = i #item['time_focus_sec']
        if len(item["location_righthand"]) > 0:
            # print(item["location_righthand"])
            fp_image = os.path.normpath(os.path.join(dir_detection_result,item["location_righthand"]["file_name"]))
            q_list[i], scores[i] = landmark_finder(fp_image, dir_detection_result_save)
            # show image
            # img = cv2.imread(fp_image)
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
    #import pdb; pdb.set_trace()
    # length of not None
    from pyquaternion import Quaternion
    distance_array_1 = np.array(len(q_list) * [0.0])
    #import pdb; pdb.set_trace()
    def dist_q(q1, q2):
        return 2 * np.arccos(np.dot(q1.elements, q2.elements))
    for i, q in enumerate(q_list):
        if q is None:
            continue
        print(i)
        for j in range(i+1, len(q_list)):
            if q_list[j] is not None:
                distance_array_1[i] = Quaternion.absolute_distance(q, q_list[j])/(j-i)
                #import pdb; pdb.set_trace()
                break
    plt.figure()
    plt.plot(time_frame, distance_array_1)
    plt.show()


    q_list_not_none = [q for q in q_list if q is not None]
    print(len(q_list_not_none))
    from pyquaternion import Quaternion
    interporation_flag = False
    none_count = 0
    none_index_start = 0
    for i, q in enumerate(q_list):
        if q is None and i>0:
            if q_list[i-1] is not None:
                #import pdb; pdb.set_trace()
                interporation_flag = True
                none_count = 0
                none_index_start = i
            none_count += 1
        elif interporation_flag and q is not None:
            interporation_flag = False
            # interpolate
            distance = i - none_index_start + 1
            stride = 1
            #import pdb; pdb.set_trace()
            #print(i)
            #print(distance)
            for j in range(none_index_start, i):
                q_list[j] = Quaternion.slerp(q_list[none_index_start-1], q, amount=stride*1.0/distance)
                stride += 1
    #import pdb; pdb.set_trace()
    distance_array = np.array(len(q_list) * [0.0])
    #import pdb; pdb.set_trace()
    def dist_q(q1, q2):
        return 2 * np.arccos(np.dot(q1.elements, q2.elements))
    for i, q in enumerate(q_list):
        if i == len(q_list)-1:
            break
        if q is not None and q_list[i+1] is not None:
            distance_array[i] = Quaternion.absolute_distance(q_list[i], q_list[i+1])
    # plot
    plt.figure()
    plt.plot(time_frame, distance_array)
    plt.figure()
    plt.plot(time_frame, scores)
    plt.show()
    #for i, loc in enumerate(json_data):
    #    if self.hand_laterality == 'right':
    #        handloc = loc['location_righthand']
    #    else:
    #        handloc = loc['location_lefthand']
    #    # crop hand images
    #    if len(handloc) > 0:
    #        frame_crop_first_frame = self.first_frame[
    #            handloc['top']:handloc['bottom'],
    #            handloc['left']:handloc['right']]
    #        tmp_focus_frame = frame_focus[i]
    #        cap.set(cv2.CAP_PROP_POS_FRAMES, tmp_focus_frame)
    #        _, frame = cap.read()
    #        frame_depth = depth_array[tmp_focus_frame]
    #        # need to reshape the depth to be exactly the same as the video
    #        frame_depth = cv2.resize(
    #            frame_depth,
    #            (self.width,
    #                self.height),
    #            interpolation=cv2.INTER_NEAREST)
    #        frame = cv2.resize(
    #            frame,
    #            (self.width, self.height),
    #            interpolation=cv2.INTER_NEAREST)
    #        fp_loc, _ = localizer.run(
    #            frame, frame_depth, handloc, self.output_dir, "hand_" + str(i))
    #        if fp_loc is not None:
    #            with open(fp_loc) as f:
    #                json_data = json.load(f)
    #                loc_xyz = json_data['roi_position']
    #                loc_xyz_list.append(self.transform(loc_xyz))
    #np_loc_xyz_list = np.array(loc_xyz_list)
    #np_loc_xyz_list = np.reshape(np_loc_xyz_list, (-1, 3))
    #np.savetxt(
    #    os.path.join(
    #        self.output_dir,
    #        "trajectory.csv"),
    #    np_loc_xyz_list,
    #    delimiter=',')
