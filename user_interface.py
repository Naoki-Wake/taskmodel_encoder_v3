
import numpy as np
import PySimpleGUI as sg
import cv2
from pathlib import Path
import json
import os
import pickle
import glob
import open3d as o3d
import utils.mkv2mp4 as mkv2mp4
import utils.video_segmentator as video_segmentator
import utils.task_recognizer as task_recognizer
import utils.task_compiler as task_compiler
import utils.speech_recognizer as speech_recognizer
import utils.recorder_gui as recorder_gui
import asyncio
# note: base code for GUI: https://qiita.com/_yane/items/1bf052e7213ae219a453


def compile_task(
        task,
        verbal_input,
        fp_mp4,
        fp_depth_npy,
        segment_timings_frame,
        output_dir_daemon,
        hand_laterality):
    daemon = task_compiler.task_daemon(
        task,
        verbal_input,
        fp_mp4,
        fp_depth_npy,
        segment_timings_frame,
        output_dir_daemon,
        hand_laterality=hand_laterality)
    daemon.set_skillparameters()
    daemon.dump_json()
    return daemon


async def run_daemon(loop,
                     task_list, verbal_input_list, fp_mp4, fp_depth_npy,
                     segment_timings_frame_list, output_dir_daemon, hand_laterality):
    sem = asyncio.Semaphore(4)

    async def run_request(task, verbal_input, fp_mp4,
                          fp_depth_npy, segment_timings_frame, output_dir_daemon, hand_laterality):
        async with sem:
            return await loop.run_in_executor(None, compile_task,
                                              task,
                                              verbal_input,
                                              fp_mp4,
                                              fp_depth_npy,
                                              segment_timings_frame,
                                              output_dir_daemon,
                                              hand_laterality)
    damon_list = [
        run_request(
            task_list[i],
            verbal_input_list[i],
            fp_mp4,
            fp_depth_npy,
            segment_timings_frame_list[i],
            output_dir_daemon,
            hand_laterality) for i in range(
            len(task_list))]
    return await asyncio.gather(*damon_list)


def file_read():
    '''
    Read the file and return the content
    '''
    layout = [
        [
            sg.Submit("Record", font='Helvetica 14')
        ],
        [
            sg.FileBrowse(key="mkvfile", font='Helvetica 14'),
            sg.Text("mkv file", font='Helvetica 14'),
            sg.InputText(font='Helvetica 14')
        ],
        [
            sg.FileBrowse(key="audiofile", font='Helvetica 14'),
            sg.Text("audio file (if any)", font='Helvetica 14'),
            sg.InputText(font='Helvetica 14')
        ],
        [sg.Submit(key="submit", font='Helvetica 14'), sg.Cancel("Exit", font='Helvetica 14')]
    ]

    window = sg.Window("file selection", layout)

    while True:
        event, values = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'Record':
            # disable this window
            window.close()
            fp_mkv, fp_audio, fp_dir = recorder_gui.run()
            window.close()
            return Path(fp_mkv), Path(fp_audio), fp_dir
        elif event == 'submit':
            if values[0] == "":
                sg.popup("no video file input")
                event = ""
            else:
                fp_mkv = values[0]
                fp_audio = values[1]
                if values[1] == "":
                    window.close()
                    return Path(fp_mkv), None, None
                break
    window.close()
    return Path(fp_mkv), Path(fp_audio), None


class Main:
    def __init__(self, fp_video, trasncript=""):
        self.fp = fp_video
        self.cap = cv2.VideoCapture(str(self.fp))
        self.transcript = trasncript

        self.ret, self.f_frame = self.cap.read()
        if self.ret:

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_count = 0
            self.s_frame = 0
            self.e_frame = self.total_count
            self.stop_flg = False
            cv2.namedWindow("Movie")
        else:
            sg.Popup("load error")
            return

    def run(self):
        layout = [
            [
                sg.Text("Start", size=(8, 1), font='Helvetica 14'),
                sg.Slider(
                    (0, self.total_count - 1),
                    0,
                    1,
                    orientation='h',
                    size=(45, 15),
                    key='-START FRAME SLIDER-',
                    enable_events=True,
                    font='Helvetica 14'
                )
            ],
            [
                sg.Text("End ", size=(8, 1), font='Helvetica 14'),
                sg.Slider(
                    (0, self.total_count - 1), self.total_count - 1,
                    1,
                    orientation='h',
                    size=(45, 15),
                    key='-END FRAME SLIDER-',
                    enable_events=True,
                    font='Helvetica 14'

                )
            ],
            [sg.Slider(
                (0, self.total_count - 1),
                0,
                1,
                orientation='h',
                size=(50, 15),
                key='-PROGRESS SLIDER-',
                enable_events=True,
                font='Helvetica 14'
            )],
            [
                sg.Button('<<<', size=(5, 1), font='Helvetica 14'),
                sg.Button('<<', size=(5, 1), font='Helvetica 14'),
                sg.Button('<', size=(5, 1), font='Helvetica 14'),
                sg.Button('Play / Stop', size=(9, 1), font='Helvetica 14'),
                sg.Button('Reset', size=(7, 1), font='Helvetica 14'),
                sg.Button('>', size=(5, 1), font='Helvetica 14'),
                sg.Button('>>', size=(5, 1), font='Helvetica 14'),
                sg.Button('>>>', size=(5, 1), font='Helvetica 14')
            ],
            [
                sg.Text("Speed", size=(6, 1), font='Helvetica 14'),
                sg.Slider(
                    (0, 240),
                    10,
                    10,
                    orientation='h',
                    size=(19.4, 15),
                    key='-SPEED SLIDER-',
                    enable_events=True,
                    font='Helvetica 14'
                ),
                sg.Text("Skip", size=(6, 1), font='Helvetica 14'),
                sg.Slider(
                    (0, 300),
                    0,
                    1,
                    orientation='h',
                    size=(19.4, 15),
                    key='-SKIP SLIDER-',
                    enable_events=True,
                    font='Helvetica 14'
                )
            ],
            [sg.HorizontalSeparator()],
            [
                sg.Text("Verbal input", font='Helvetica 14'),
                sg.InputText(self.transcript, key='-VERBAL-', font='Helvetica 14'),
                sg.Submit("Confirm", key="Confirm", font='Helvetica 14')
            ],
        ]
        window = sg.Window('Video confirmation', layout, location=(0, 0))

        self.event, values = window.read(timeout=0)
        print("File loaded successfully")
        print("File Path: " + str(self.fp))
        print("fps: " + str(int(self.fps)))
        print("width: " + str(self.width))
        print("height: " + str(self.height))
        print("frame count: " + str(int(self.total_count)))
        # main loop
        try:
            while True:
                self.event, values = window.read(
                    timeout=values["-SPEED SLIDER-"]
                )

                if self.event != "__TIMEOUT__":
                    print(self.event)

                # Exit condition
                if self.event in ('Exit', sg.WIN_CLOSED, None):
                    self.transcript = ""
                    break
                if self.event in ('Confirm'):
                    print(values['-VERBAL-'])
                    verbal_input = values['-VERBAL-']
                    self.transcript = verbal_input
                    break

                if self.event == 'Reset':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    continue

                if self.event == '-PROGRESS SLIDER-':
                    self.frame_count = int(values['-PROGRESS SLIDER-'])
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
                    if values['-PROGRESS SLIDER-'] > values['-END FRAME SLIDER-']:
                        window['-END FRAME SLIDER-'].update(
                            values['-PROGRESS SLIDER-'])

                if self.event == '-START FRAME SLIDER-':
                    self.s_frame = int(values['-START FRAME SLIDER-'])
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    if values['-START FRAME SLIDER-'] > values['-END FRAME SLIDER-']:
                        window['-END FRAME SLIDER-'].update(
                            values['-START FRAME SLIDER-'])
                        self.e_frame = self.s_frame

                if self.event == '-END FRAME SLIDER-':
                    if values['-END FRAME SLIDER-'] < values['-START FRAME SLIDER-']:
                        window['-START FRAME SLIDER-'].update(
                            values['-END FRAME SLIDER-'])
                        self.s_frame = self.e_frame
                    self.e_frame = int(values['-END FRAME SLIDER-'])

                if self.event == '<<<':
                    self.frame_count = np.maximum(0, self.frame_count - 150)
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

                if self.event == '<<':
                    self.frame_count = np.maximum(0, self.frame_count - 30)
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

                if self.event == '<':
                    self.frame_count = np.maximum(0, self.frame_count - 1)
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

                if self.event == '>':
                    self.frame_count = self.frame_count + 1
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

                if self.event == '>>':
                    self.frame_count = self.frame_count + 30
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

                if self.event == '>>>':
                    self.frame_count = self.frame_count + 150
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

                # Loop video
                if self.frame_count >= self.e_frame:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    continue

                if self.event == 'Play / Stop':
                    self.stop_flg = not self.stop_flg

                if((self.stop_flg and self.event == "__TIMEOUT__")):
                    window['-PROGRESS SLIDER-'].update(self.frame_count)
                    continue

                if not self.stop_flg and values['-SKIP SLIDER-'] != 0:
                    self.frame_count += values["-SKIP SLIDER-"]
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

                self.ret, self.frame = self.cap.read()
                self.valid_frame = int(self.frame_count - self.s_frame)

                if not self.ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                    self.frame_count = self.s_frame
                    continue

                cv2.putText(self.frame,
                            str("framecount: {0:.0f}".format(self.frame_count)),
                            (15,
                             20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (240,
                                230,
                                0),
                            1,
                            cv2.LINE_AA)
                cv2.putText(self.frame,
                            str("time: {0:.1f} sec".format(self.frame_count / self.fps)),
                            (15,
                                40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (240,
                                230,
                                0),
                            1,
                            cv2.LINE_AA)

                cv2.imshow("Movie", self.frame)
                if self.stop_flg:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
                else:
                    self.frame_count += 1
                    window['-PROGRESS SLIDER-'].update(self.frame_count + 1)

        finally:
            cv2.destroyWindow("Movie")
            self.cap.release()
            window.close()
            return self.transcript


if __name__ == '__main__':
    debug = True
    # get file paths
    fp_mkv, fp_audio, output_dir_name_root = file_read()
    if output_dir_name_root is None:
        # output_dir_name_root = "output_user_interface"
        output_dir_name_root = os.path.dirname(fp_mkv)
        # print(output_dir_name_root)
    filename_mkv = os.path.basename(fp_mkv)
    output_dir = os.path.join(
        os.getcwd(),
        output_dir_name_root,
        filename_mkv.split('.')[0],
        'task_recognition')
    output_dir_daemon = os.path.join(
        os.getcwd(),
        output_dir_name_root,
        filename_mkv.split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        debug = False

    # extract mkv file
    print('extracting mkv...')
    if debug:
        fp_mp4 = os.path.join(output_dir, filename_mkv.split('.')[0] + '.mp4')
        fp_depth_npy = os.path.join(
            output_dir,
            filename_mkv.split('.')[0] +
            '_depth.npy')
        if not os.path.exists(fp_mp4) or not os.path.exists(fp_depth_npy):
            debug = False
    if not debug:
        fp_mp4, fp_depth_mp4, fp_depth_npy = mkv2mp4.run(
            str(fp_mkv), output_dir, scale=1.0)
    print('done')

#    import pdb ; pdb.set_trace()
    # segment video
    print('segmenting video...')
    if debug:
        # find a file path to 'segment.json'
        fp_segmentation = os.path.join(output_dir,
                                       filename_mkv.split('.')[0]+'_rescale',
                                       filename_mkv.split('.')[0]+'_rescale',
                                       'segment.json')
        if not os.path.exists(fp_segmentation):
            debug = False
    if not debug:
        fp_segmentation = video_segmentator.run(fp_mp4, scale=0.5)
    if os.path.exists(fp_segmentation):
        with open(fp_segmentation, 'r') as f:
            segment_data = json.load(f)
            segment_timings = segment_data['frame_minimum']
            segment_timings_sec = segment_data['time_minimum']
    timeparts_frame = []
    timeparts_sec = []
    for i in range(len(segment_timings) - 1):
        timeparts_frame.append(
            (segment_timings[i], segment_timings[i + 1]))
        timeparts_sec.append(
            (segment_timings_sec[i], segment_timings_sec[i + 1]))
    print('done')
#    import pdb ; pdb.set_trace()
    # split videos based on timepart
    print('splitting videos...')
    if debug:
        fp_splitvideo = glob.glob(os.path.join(output_dir, 'segment_part*'))
        if len(fp_splitvideo) == 0:
            debug = False
    if not debug:
        cap = cv2.VideoCapture(fp_mp4)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # print(timeparts_frame)
        # print(output_dir)
        fp_splitvideo = [
            os.path.join(
                output_dir,
                f"segment_part_{start}-{end}.mp4") for start,
            end in timeparts_frame]
        writers = [cv2.VideoWriter(fp_tmp, fourcc, 30.0, (w, h))
                   for fp_tmp in fp_splitvideo]
        # print(fp_splitvideo)
        f = 0
        while ret:
            f += 1
            for i, part in enumerate(timeparts_frame):
                start, end = part
                if start <= f <= end:
                    writers[i].write(frame)
            ret, frame = cap.read()
        for writer in writers:
            writer.release()
        cap.release()
    print('done')
#    import pdb ; pdb.set_trace()
    # audio file
    print('analyzing audios...')
    audio_data = {}
    transcript = []

    if fp_audio is not None:
        if debug:
            fp_text = os.path.join(
                output_dir,
                'speech_recognized',
                'transcript.json')
            if not os.path.exists(fp_text):
                debug = False
        if not debug:
            fp_text = speech_recognizer.run(
                fp_audio, fp_segmentation, output_dir)
        if os.path.exists(fp_text):
            with open(fp_text, 'r') as f:
                audio_data = json.load(f)
                transcript = audio_data['recognized_text']
    print('done')
#    import pdb ; pdb.set_trace()
    # confirm the verbal input
    print('confirming the verbal input...')
    transcript_confirmed = []
    if debug:
        fp_text_confirmed = os.path.join(
            output_dir, 'transcript_confirmed.json')
        if not os.path.exists(fp_text_confirmed):
            debug = False
    if not debug:
        for i, fp_video_item in enumerate(fp_splitvideo):
            transcript_item = ""
            if i < len(transcript):
                transcript_item = transcript[i]
            confirmed_transcript_item = Main(
                fp_video_item, transcript_item).run()
            transcript_confirmed.append(confirmed_transcript_item)
        dump = {}
        dump['recognized_text'] = transcript_confirmed
        dump['fp_video'] = fp_splitvideo
        dump['segment_timings_frame'] = timeparts_frame
        dump['segment_timings_sec'] = timeparts_sec
        fp_text_confirmed = os.path.join(
            output_dir, "transcript_confirmed.json")
        with open(fp_text_confirmed, 'w') as f:
            json.dump(dump, f, indent=4)

    with open(fp_text_confirmed, 'r') as f:
        data = json.load(f)
        transcript_confirmed = data['recognized_text']
        fp_video = data['fp_video']
        timeparts_frame = data['segment_timings_frame']
        timeparts_sec = data['segment_timings_sec']
    print('done')

    # Write the result of segmentation after user's confirmation
    print('writing video...')
    if not debug:
        parts_confirmed = []
        parts_confirmed_show = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for i, part in enumerate(transcript_confirmed):
            if part != "":
                parts_confirmed.append(timeparts_frame[i])
                parts_confirmed_show.append(part)
        cap = cv2.VideoCapture(fp_mp4)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        writer = cv2.VideoWriter(
            os.path.join(
                output_dir, 'segment_result.mp4'), fourcc, 30.0, (w, h))
        f = 0
        while ret:
            f += 1
            currentseg = None
            for i, part in enumerate(parts_confirmed):
                start, end = part
                if start <= f <= end:
                    currentseg = i
            # draw currentseg info to the frame
            if currentseg is not None:
                # font size = 5
                cv2.putText(
                    frame,
                    f"Segment index: {currentseg}",
                    (10,
                        60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    4,
                    color=(
                        30,
                        30,
                        255),
                    thickness=4)
            else:
                cv2.putText(
                    frame,
                    f"Not manipulation",
                    (10,
                        60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    4,
                    color=(
                        30,
                        30,
                        255),
                    thickness=4)
            writer.write(frame)
            ret, frame = cap.read()
        writer.release()
        cap.release()
    print('done')

    print('encoding task model...')
    if debug:
        fp_daemon = os.path.join(output_dir, "daemons.pkl")
        if not os.path.exists(fp_daemon):
            debug = False
    if not debug:
        #  recognize tasks
        print('recognizing tasks...')
        fp_task_recognized = task_recognizer.run_modelbased(
            fp_text_confirmed, output_dir)
        # TODO consistency check and review the task sequence
        print('done')

        print('Checking hand laterality...')
        hand_laterality = ''
        with open(fp_task_recognized, 'r') as f:
            task_recognized = json.load(f)
        for i, task in enumerate(task_recognized["recognized_tasks"]):
            segment_timings_frame = task_recognized["segment_timings_frame"][i]
            verbal_input = task_recognized["recognized_text"][i]
            if task == "GRASP":
                print("encoding: " + task)
                daemon = task_compiler.task_daemon(
                    task,
                    verbal_input,
                    fp_mp4,
                    fp_depth_npy,
                    segment_timings_frame,
                    output_dir_daemon,
                    hand_laterality='unknown')
                hand_laterality = daemon.hand_laterality
        print(hand_laterality)
        print('done')

        print('compiling task models...')
        # serialize the task sequence
        daemons = []
        manipulation_flag = False
        with open(fp_task_recognized, 'r') as f:
            task_recognized = json.load(f)

        task_list = []
        verbal_input_list = []
        segment_timings_frame_list = []
        for i, task in enumerate(task_recognized["recognized_tasks"]):
            segment_timings_frame = task_recognized["segment_timings_frame"][i]
            verbal_input = task_recognized["recognized_text"][i]
            if task == "NOTTASK" or task == "UNKNOWN":
                continue
            else:
                print("encoding: " + task)
                print(segment_timings_frame)
                task_list.append(task)
                verbal_input_list.append(verbal_input)
                segment_timings_frame_list.append(segment_timings_frame)
        loop = asyncio.get_event_loop()
        # edit task model sequence to start with STG12->grasp and end with
        # release->STG12
        # TODO we might need more sphisticated rule
        valid_task_list_i = []
        manipulation_flag = False
        for i, item in enumerate(task_list):
            if manipulation_flag == False and item == "GRASP":
                manipulation_flag = True
                if i > 0 and task_list[i - 1] == "PTG12":
                    valid_task_list_i.append(i - 1)
            if manipulation_flag:
                valid_task_list_i.append(i)
            if manipulation_flag and item == "RELEASE":
                manipulation_flag = False
        valid_task_list_i.sort()
        task_list = [task_list[i] for i in valid_task_list_i]
        verbal_input_list = [verbal_input_list[i] for i in valid_task_list_i]
        segment_timings_frame_list = [
            segment_timings_frame_list[i] for i in valid_task_list_i]

        daemons = loop.run_until_complete(
            run_daemon(
                loop,
                task_list,
                verbal_input_list,
                fp_mp4,
                fp_depth_npy,
                segment_timings_frame_list,
                output_dir_daemon,
                hand_laterality))

        fp_daemon = os.path.join(output_dir, "daemons.pkl")
        with open(fp_daemon, 'wb') as f:
            pickle.dump(daemons, f)
    print('done')

    fp_daemon = os.path.join(output_dir, "daemons.pkl")
    with open(fp_daemon, 'rb') as f:
        daemons = pickle.load(f)

    # concatenate the task sequence
    task_models = []
    for daemon in daemons:
        task_models.append(daemon.taskmodel_json)

    task_models_save = []
    # manually modify the task sequence (parameter filling)
    for i, item in enumerate(task_models):
        if i > 0 and item["_task"] == "GRASP":
            item_pre = task_models[i - 1]
            # task_models_save.append(item_pre)
            item["prepre_grasp_position"]["value"] = item_pre["start_position"]["value"]

    # manually modify the task sequence (trim the task sequence)
    # manipulation_flag = False
    # for item in task_models:
    #     if item["_task"] == "GRASP":
    #         manipulation_flag = True
    #     if manipulation_flag:
    #         task_models_save.append(item)
    #     if item["_task"] == "RELEASE":
    #         manipulation_flag = False
    # encode all the found tasks
    for item in task_models:
        task_models_save.append(item)

    # save the task sequence
    task_models_save_json = {}
    task_models_save_json["version"] = "1.0"
    task_models_save_json["rawdata_path"] = str(fp_mkv)
    task_models_save_json["task_models"] = task_models_save
    fp_task_sequence = os.path.join(output_dir, "task_models.json")
    with open(fp_task_sequence, 'w') as f:
        json.dump(task_models_save_json, f, indent=4)

    def circle_fitting(xi, yi):
        M = np.array([[np.sum(xi ** 2), np.sum(xi * yi), np.sum(xi)],
                      [np.sum(xi * yi), np.sum(yi ** 2), np.sum(yi)],
                      [np.sum(xi), np.sum(yi), 1 * len(xi)]])
        Y = np.array([[-np.sum(xi ** 3 + xi * yi ** 2)],
                      [-np.sum(xi ** 2 * yi + yi ** 3)],
                      [-np.sum(xi ** 2 + yi ** 2)]])

        M_inv = np.linalg.inv(M)
        X = np.dot(M_inv, Y)
        a = - X[0] / 2
        b = - X[1] / 2
        r = np.sqrt((a ** 2) + (b ** 2) - X[2])
        return a, b, r

    def generate_circle_points(a, b, r, dtheta):
        # generate points on the circle
        theta = np.arange(0, 2 * np.pi, dtheta)
        x = a + r * np.cos(theta)
        y = b + r * np.sin(theta)
        return x, y
    # concatenate the hand points
    if len(daemons) > 0:
        pcd_list = []
        if daemons[0].fp_3dmodel_hand_first is not None:
            pcd = o3d.io.read_point_cloud(daemons[0].fp_3dmodel_hand_first)
            pcd.remove_non_finite_points()
            pcd.uniform_down_sample(7)
            # transform the coordinates to the original image
            if daemons[0].rot_mat_4x4_marker_to_camera is not None:
                pcd.transform(daemons[0].rot_mat_4x4_marker_to_camera)
            pcd_list.append(pcd)
        for daemon in daemons:
            if daemon.fp_3dmodel_hand_last is not None:
                pcd = o3d.io.read_point_cloud(daemon.fp_3dmodel_hand_last)
                pcd.remove_non_finite_points()
                pcd.uniform_down_sample(7)
                # transform the coordinates to the original image
                if daemon.rot_mat_4x4_marker_to_camera is not None:
                    pcd.transform(daemon.rot_mat_4x4_marker_to_camera)
                pcd_list.append(pcd)
            if daemon.task == "GRASP":
                if daemon.fp_3dmodel_object_first is not None:
                    pcd = o3d.io.read_point_cloud(
                        daemon.fp_3dmodel_object_first)
                    pcd.remove_non_finite_points()
                    pcd.uniform_down_sample(7)
                    # transform the coordinates to the original image
                    if daemon.rot_mat_4x4_marker_to_camera is not None:
                        pcd.transform(daemon.rot_mat_4x4_marker_to_camera)
                    pcd_list.append(pcd)
                    pcd_list.append(pcd)
            if daemon.task == "PTG5":
                hand_position = daemon.taskmodel_json["hand_trajectory"]["value"]
                for point in hand_position:
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.005)
                    mesh.translate(tuple(point))
                    mesh.paint_uniform_color([1, 0, 0])
                    pcd_list.append(mesh)
                hand_position = np.array(hand_position)
                xi = hand_position[:, 0]
                yi = hand_position[:, 1]
                # a, b, r = circle_fitting(xi, yi)
                rotation_center_position = daemon.taskmodel_json["rotation_center_position"]["value"]
                a = rotation_center_position[0]
                b = rotation_center_position[1]
                r = daemon.taskmodel_json["rotation_radius"]["value"]
                xi, yi = generate_circle_points(a, b, r, np.pi / 20.0)
                for x, y in zip(xi, yi):
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.005)
                    mesh.translate((x, y, rotation_center_position[2]))
                    mesh.paint_uniform_color([0, 1, 0])
                    pcd_list.append(mesh)
        o3d.visualization.draw_geometries(pcd_list)

    if len(daemons) > 0:
        pcd_list = []
        for daemon in daemons:
            if daemon.task == "PTG5":
                fp_first, _ = daemons[0]._extract_pointcloud()
                pcd = o3d.io.read_point_cloud(fp_first)
                pcd.remove_non_finite_points()
                pcd.uniform_down_sample(15)
                # transform the coordinates to the original image
                if daemons[0].rot_mat_4x4_marker_to_camera is not None:
                    pcd.transform(daemons[0].rot_mat_4x4_marker_to_camera)
                pcd_list.append(pcd)

                _, fp_last = daemons[-1]._extract_pointcloud()
                pcd = o3d.io.read_point_cloud(fp_last)
                pcd.remove_non_finite_points()
                pcd.uniform_down_sample(15)
                # transform the coordinates to the original image
                if daemons[-1].rot_mat_4x4_marker_to_camera is not None:
                    pcd.transform(daemons[-1].rot_mat_4x4_marker_to_camera)
                pcd_list.append(pcd)

                hand_position = daemon.taskmodel_json["hand_trajectory"]["value"]
                for point in hand_position:
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.005)
                    mesh.translate(tuple(point))
                    mesh.paint_uniform_color([1, 0, 0])
                    pcd_list.append(mesh)
                hand_position = np.array(hand_position)
                xi = hand_position[:, 0]
                yi = hand_position[:, 1]
                a, b, r = circle_fitting(xi, yi)
                xi, yi = generate_circle_points(a, b, r, np.pi / 20.0)
                for x, y in zip(xi, yi):
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.005)
                    mesh.translate((x, y, np.median(hand_position[:, 2])))
                    mesh.paint_uniform_color([0, 1, 0])
                    pcd_list.append(mesh)

                pcd = o3d.io.read_point_cloud(daemon.fp_3dmodel_hand_last)
                pcd.remove_non_finite_points()
                pcd.uniform_down_sample(7)
                # transform the coordinates to the original image
                if daemon.rot_mat_4x4_marker_to_camera is not None:
                    pcd.transform(daemon.rot_mat_4x4_marker_to_camera)
                pcd_list.append(pcd)
                pcd = o3d.io.read_point_cloud(daemon.fp_3dmodel_hand_first)
                pcd.remove_non_finite_points()
                pcd.uniform_down_sample(7)
                # transform the coordinates to the original image
                if daemon.rot_mat_4x4_marker_to_camera is not None:
                    pcd.transform(daemon.rot_mat_4x4_marker_to_camera)
                pcd_list.append(pcd)
        o3d.visualization.draw_geometries(pcd_list)

    # TODO coordinate emply skill slots
    # TODO dump the coordinated task models as json
