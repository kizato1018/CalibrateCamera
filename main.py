import cv2
import numpy as np
import queue
import threading
from multiprocessing import Process, Manager, Event
import time
import os
import shutil
import glob
import tqdm
import json
import sys
import asyncio
from collections import deque
from enum import Enum
from functools import reduce
os.environ["PYLON_CAMEMU"] = "3"

TEST_CAMERA = True

if not TEST_CAMERA:
    from pypylon import genicam
    from pypylon import pylon


BASE_DIR = '.'
IMAGE_DIR = 'result5.28_2'
OUTPUT_DIR = 'calibrate_output5.28_2'
CAMERA_COUNT = 5
IMG_HEIGHT = 540
IMG_WIDTH = 720
CHECKERBOARD_SIZE = (4, 3)
SQUARE_SIZE = 200
START_CAMERA = 0
FPS = 20
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

class Mode(Enum):
    INTRINSIC=1
    EXTRINSIC=2
    TEST=3

def test_capture_images(cameras_queue, stop):
    cap = cv2.VideoCapture(0)
    i = 0
    last_time = 0
    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - last_time < 1/FPS/CAMERA_COUNT:
            print("skip frame.")
            continue
        last_time = time.time()
        center = (frame.shape[1]//2, frame.shape[0]//2)
        # frame = cv2.flip(frame, 1)
        frame = frame[center[1]-IMG_HEIGHT//2:center[1]+IMG_HEIGHT//2, center[0]-IMG_WIDTH//2:center[0]+IMG_WIDTH//2]

        cameras_queue[i%CAMERA_COUNT].put(frame)
        i += 1

    cap.release()
    print("Stop capture_images.")

def capture_images(cameras_queue, stop):
    maxCamerasToUse = CAMERA_COUNT
    try:

        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        print(devices)
        # save_config(tlFactory, devices)

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))


        cam_cnt = cameras.GetSize()

        # Create and attach all Pylon Devices.
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            # pylon.FeaturePersistence.Load(nodeFile, cam.GetNodeMap(), True)

            # Print the model name of the camera.
            print("Using device ", cam.GetDeviceInfo().GetModelName())
        
        cameras.Open()
        for camera in cameras:
            print(camera.PixelFormat.Value)
            camera.PixelFormat.SetValue("BayerRG8")
            camera.GevIEEE1588.Value = True
            camera.SyncFreeRunTimerEnable.SetValue(True)
            camera.SyncFreeRunTimerTriggerRateAbs.SetValue(FPS)
            fps = camera.SyncFreeRunTimerTriggerRateAbs.Value
            # countOfImagesToGrab = int(fps * record_duraion)
        for camera in cameras:
            camera.GevIEEE1588DataSetLatch.Execute()
            time.sleep(0.1)
        for camera in cameras:
            camera.SyncFreeRunTimerUpdate.Execute()
        # Starts grabbing for all cameras starting with index 0. The grabbing
        # is started for one camera after the other. That's why the images of all
        # cameras are not taken at the same time.
        # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
        # According to their default configuration, the cameras are
        # set up for free-running continuous acquisition.
        cameras.StartGrabbing(pylon.GrabStrategy_LatestImages)

        # Grab c_countOfImagesToGrab from the cameras.
        # while True:

        # images = dict()
        # for i in range(cam_cnt):
        #     images[i] = []
        print("Recording...")
        try:
            while not stop.is_set():
            # while True:
                if not cameras.IsGrabbing():
                    break
                    pass
                grabResult = cameras.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
                # print(grabResult)

                # When the cameras in the array are created the camera context value
                # is set to the index of the camera in the array.
                # The camera context is a user settable value.
                # This value is attached to each grab result and can be used
                # to determine the camera that produced the grab result.
                cameraContextValue = grabResult.GetCameraContext()

                # Print the index and the model name of the camera.
                # print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())

                # Now, the image data can be processed.
                # print("GrabSucceeded: ", grabResult.GrabSucceeded())
                # print("SizeX: ", grabResult.GetWidth())
                # print("SizeY: ", grabResult.GetHeight())
                if grabResult.GrabSucceeded():
                    img = grabResult.GetArray()
                    image = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
                    cameras_queue[cameraContextValue].put(image)
                # print(f"{cameraContextValue}: {(i >> 1):06d} {((i >> 1) / fps):.2f}")
                # cv2.imwrite(f"output/camera{cameraContextValue}/img{i>>1}.png", img)
                # cv2.imshow(f"camera{cameraContextValue}", img)
                # k = cv2.waitKey(1)
                    # print("Gray value of first pixel: ", img[0, 0])
        except KeyboardInterrupt:
            pass
        print("Record Done.")

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e)
        exitCode = 1

async def Undistortion(images, output_dir, mtx, dist):
    output_dir = os.path.join(output_dir, 'undistorted_images')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    for frame, image in tqdm.tqdm(enumerate(images)):
        img = cv2.imread(image)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(f'{output_dir}/undistorted{frame}.png', dst)

def compute_intrinsic(calibrate_return, shared_mtx, shared_dist, camera_idx):
    images_path = sorted(glob.glob(f'{BASE_DIR}/{IMAGE_DIR}/calibrate{camera_idx}/camera{camera_idx}/*.png'))
    if len(images_path) == 0:
        print(f"No images to calibrate intrinsic for {camera_idx}.")
        # return False, None, None
        calibrate_return.value = 0
        return

    # Arrays to store object points and image points
    object_points = []  
    image_points = []
    output_dir = os.path.join(BASE_DIR, OUTPUT_DIR, f"camera{camera_idx}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for frame, image in tqdm.tqdm(enumerate(images_path)):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
        
        if ret:
            # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            object_points.append(objp)
            image_points.append(corners)
            out_img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, ret)
            cv2.imwrite(f'{output_dir}/chessboard{frame}.png', out_img)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (IMG_WIDTH, IMG_HEIGHT), None, None)
    
    asyncio.run(Undistortion(images_path, output_dir, mtx, dist))

     # Save intrinsic parameters as JSON
    fisheye_param = {
        "class_name": "FisheyeCameraParameter",
        "convention": "opencv",
        "height": IMG_HEIGHT,
        "width": IMG_WIDTH,
        "intrinsic": mtx.tolist(),
        "k1": dist[0][0],
        "k2": dist[0][1],
        "k3": dist[0][4],
        "k4": 0.0,
        "k5": 0.0,  # Additional parameters, if you don't have them set to 0
        "k6": 0.0,
        "p1": dist[0][2],
        "p2": dist[0][3],
        "world2cam": True,
        "extrinsic_r": [[0,0,0],[0,0,0],[0,0,0]],
        "extrinsic_t": [0,0,0]
    }
    fisheye_param_path = os.path.join(output_dir, f"fisheye_param_{camera_idx:02}.json")
    with open(fisheye_param_path, 'w') as f:
        json.dump(fisheye_param, f, indent=4)

    calibrate_return.value = 1
    shared_mtx.extend(mtx)
    shared_dist.extend(dist)
    
    # return True, mtx, dist

def find_chessboards(path:str,
                     frame_id:int, 
                     frame, 
                     is_found,
                     found_count:int, 
                     grid, 
                     percent:float=0.8,
                     save_fps:int=FPS, 
                     calibrate_queue=None, 
                     mode:Mode = Mode.TEST):
    done = False
    camera_idx = int(path[-1])
    chessboard_corners = None
    grid_height, grid_width = grid.shape[:2]
    target_corners = np.array([[(1-percent) * frame.shape[1], (1-percent) * frame.shape[0]], [percent * frame.shape[1], (1-percent) * frame.shape[0]], [percent * frame.shape[1], percent * frame.shape[0]], [(1-percent) * frame.shape[1], percent * frame.shape[0]]], dtype=np.int32)
    original_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None, cv2.CALIB_CB_FAST_CHECK)
    if mode == Mode.EXTRINSIC:
        is_found[frame_id * (CAMERA_COUNT + 1) + camera_idx] = ret
        # if frame_id % 5 == 0 and camera_idx == 0:
        #     print("before is_found: ", is_found[frame_id * (CAMERA_COUNT + 1) + CAMERA_COUNT])
        is_found[frame_id * (CAMERA_COUNT + 1) + CAMERA_COUNT] += 1
        # if frame_id % 5 == 0 and camera_idx == 0:
        #     print("after is_found: ", is_found[frame_id * (CAMERA_COUNT + 1) + CAMERA_COUNT])

    x, y = -1, -1
    padding = 2
    grid_width_size = (target_corners[1][0] - target_corners[0][0]) // grid_width
    grid_height_size = (target_corners[2][1] - target_corners[1][1]) // grid_height
    if grid[0, 0, 0] == 0:
        grid[:, :, 0] = np.array([[target_corners[0][0] + i * grid_width_size + padding for i in range(grid_width)] for _ in range(grid_height)])
        grid[:, :, 1] = np.array([[target_corners[0][1] + i * grid_height_size + padding for _ in range(grid_width)] for i in range(grid_height)])
        grid[:, :, 2] = np.array([[target_corners[0][0] + (i+1) * grid_width_size for i in range(grid_width)] for _ in range(grid_height)])
        grid[:, :, 3] = np.array([[target_corners[0][1] + (i+1) * grid_height_size for _ in range(grid_width)] for i in range(grid_height)])

    # 畫出 Grid
    for i in range(grid_width):
        for j in range(grid_height):
            if grid[j, i, 4] == 0:
                cv2.rectangle(frame, (grid[j, i, 0], grid[j, i, 1]), (grid[j, i, 2], grid[j, i, 3]), (100, 100, 100), 2)
            else:
                cv2.rectangle(frame, (grid[j, i, 0], grid[j, i, 1]), (grid[j, i, 2], grid[j, i, 3]), (0, 255, 0), 2)



    # 畫從中心向外80%的範圍矩形
    center = (frame.shape[1]//2, int(frame.shape[0] * 0.66))
    cv2.circle(frame, center, 10, (255), 3)


   
   
    
    if ret:
        # print("Found chessboard in camera {}".format(camera_idx))
        # 將找到的方格板圖像加入到 found_images 中
        chessboard_corners = np.array(corners.reshape(-1, 2), dtype=np.int32)
        chessboard_center = np.mean(chessboard_corners, axis=0, dtype=np.int32)
        cv2.circle(frame, tuple(chessboard_center), 5, (0, 0, 255), -1)

        # check which grid the chessboard is in
        if grid[0, 0, 0] < chessboard_center[0] < grid[-1, -1, 2] and grid[0, 0, 1] < chessboard_center[1] < grid[-1, -1, 3]:
            x = (chessboard_center[0] - grid[0,0,0]) // grid_width_size 
            y = (chessboard_center[1] - grid[0,0,1]) // grid_height_size
        else:
            x, y = -1, -1
        
        if mode == Mode.INTRINSIC and int(path[-1]) == calibrate_queue[0]:
            if x != -1 and y != -1 and grid[y, x, 4] == 0:
                cv2.imwrite(f"{path}/img{found_count[0]:06d}.png", original_frame)
                grid[y, x, 4] = 1
                found_count[0] += 1
        if mode == Mode.EXTRINSIC:
            if found_count[0] * save_fps <= frame_id:
                fid = frame_id // save_fps + frame_id % save_fps
                cv2.imwrite(f"{path}/img{fid:06d}.png", original_frame)
                found_count[0] += 1
        cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners, ret)
    done = all(grid[:,:,4].flatten())
    
    if done:
        # print("done")
        if mode == Mode.INTRINSIC and camera_idx == calibrate_queue[0]:
            calibrate_queue.pop(0)
            
    return done, frame, (x, y)

def process_camera(camera_idx, image_queue, display_queue, is_found, stop, calibrate_queue=None, mode:Mode = Mode.TEST):
    """
    mode: intrinsic, extrinsic, test
    """
    grid_width = 8
    grid_height = 6
    grid = np.zeros((grid_height, grid_width ,5), np.int32) # 5: x1, y1, x2, y2, is_found
    found_count = [0]
    frame_id = 0
    if mode == Mode.INTRINSIC:
        path = f"{BASE_DIR}/{IMAGE_DIR}/calibrate{camera_idx}/camera{camera_idx}"
    elif mode == Mode.EXTRINSIC:
        path = f"{BASE_DIR}/{IMAGE_DIR}/all_calibrate/camera{camera_idx}"
    else:
        path = ""
    if path:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=False)
    display_frame = None
    done = False
    start_calibrate = False
    calibrate_return = Manager().Value('i', -1)
    mtx = Manager().list()
    dist = Manager().list()
    show_matrix_done = False
    checkboard_center = []
    checked_frame_id = -1
    p = None

    
    while not stop.is_set():
        if image_queue.empty():
            time.sleep(0.01)
            continue
        frame = image_queue.get()
        checkboard_center_grid = None
        if len(is_found)  // (CAMERA_COUNT + 1)  < frame_id + 1:
            is_found += [False] * CAMERA_COUNT + [0]
            # print("is_found: ", is_found)
        
        if mode == Mode.INTRINSIC:
            if not done:
                done, display_frame, checkboard_center_grid = find_chessboards(path, frame_id, frame, is_found, found_count, grid, percent=0.8,save_fps=5, calibrate_queue=calibrate_queue, mode=mode)
                frame_id += 1
            if done:
                if not start_calibrate:
                    start_calibrate = True
                    p = Process(target=compute_intrinsic, args=(calibrate_return, mtx, dist, camera_idx))
                    p.start()
                if calibrate_return.value != -1:
                    if calibrate_return.value == 0:
                        print(f"Camera {camera_idx} intrinsic calibration failed.")
                    elif not show_matrix_done:
                        print(f"Camera {camera_idx} intrinsic calibration success.")
                        print(f"mtx: {mtx}")
                        print(f"dist: {dist}")
                        fx = round(mtx[0][0], 1)
                        fy = round(mtx[1][1], 1)
                        cx = round(mtx[0][2], 1)
                        cy = round(mtx[1][2], 1)
                        k1 = round(dist[0][0], 1)
                        k2 = round(dist[0][1], 1)
                        k3 = round(dist[0][4], 1)
                        p1 = round(dist[0][2], 1)
                        p2 = round(dist[0][3], 1)
                        
                        msg_start_pos = (IMG_WIDTH - 200, 30)
                        display_frame = cv2.flip(display_frame, 1)
                        for frame_id, msg in enumerate([f"fx: {fx}", f"fy: {fy}", f"cx: {cx}", f"cy: {cy}", f"k1: {k1}", f"k2: {k2}", f"k3: {k3}", f"p1: {p1}", f"p2: {p2}"]):
                            display_frame = cv2.putText(display_frame, msg, (msg_start_pos[0], msg_start_pos[1] + 30 * frame_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        display_frame = cv2.flip(display_frame, 1)
                        show_matrix_done = True

                    # reset
                    if len(calibrate_queue) > 0 and calibrate_queue[-1] == camera_idx:
                        grid = np.zeros((grid_height, grid_width ,5), np.int32)
                        done = False
                        found_count = [0]
                        frame_id = 0
                        if path:
                            if os.path.exists(path):
                                shutil.rmtree(path)
                            os.makedirs(path, exist_ok=False)
                        start_calibrate = False
                        calibrate_return.value = -1
                        mtx.clear()
                        dist.clear()
                        show_matrix_done = False
                    
                    p.join()
        elif mode == Mode.EXTRINSIC:
            done, display_frame, checkboard_center_grid = find_chessboards(path, frame_id, frame, is_found, found_count, grid, percent=0.7, save_fps=5, calibrate_queue=calibrate_queue, mode=mode)
            frame_id += 1
            if len(is_found) > 12 and frame_id % 5 == 0 and camera_idx == 3:
                print("checked_frame_id: ", checked_frame_id)
                print("is_found: ", is_found[-12:])
            if len(is_found) // (CAMERA_COUNT + 1) > checked_frame_id + 2 and is_found[(checked_frame_id+1) * (CAMERA_COUNT + 1) + CAMERA_COUNT] == CAMERA_COUNT:
                valid_frame = reduce(lambda x, y: x + 1 if y else x, is_found[(checked_frame_id+1) * (CAMERA_COUNT + 1): (checked_frame_id+1) * (CAMERA_COUNT + 1) + CAMERA_COUNT], 0) >= 2 and is_found[(checked_frame_id+1) * (CAMERA_COUNT + 1) + camera_idx]
                if valid_frame:
                    x, y = checkboard_center_grid
                    if x != -1 and y != -1:
                        grid[y, x, 4] = True
                checked_frame_id += 1
            if done:
                msg_start_pos = (IMG_WIDTH - 200, 30)
                display_frame = cv2.flip(display_frame, 1)
                display_frame = cv2.putText(display_frame, "done.", msg_start_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                display_frame = cv2.flip(display_frame, 1)
        display_queue.put(display_frame)
    print("Stop process_camera. camera_idx: ", camera_idx)

def display_images(display_queue, stop, calibrate_queue):
    current_fps = 0
    last_time = time.time()
    window_title = "Multi-camera Display"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    frame_idx = 0
    while not stop.is_set():
        if frame_idx % 5 == 0:
            print(f"calibrate_queue: {calibrate_queue}")
        # print(len(cameras[0]), len(cameras[1]), len(cameras[2]))
        # if all(cameras[i][idx] is not None for i in range(num_cameras)):
        check_count = 0
        # if all(not camera.empty() for camera in display_queue):
        frames = []
        for camera in display_queue:
            frame = camera.get()
            frame = np.flip(frame, axis=1)
            frames.append(frame)
        if len(frames) >= 3:
            columns = 3
            if len(frames) % columns != 0:
                padding = len(frames) % columns
                frames += [np.zeros_like(frames[0])] * padding
            rows = len(frames) // columns
            hframes = []
            for i in range(rows):
                hframes.append(np.hstack(frames[i*3:(i+1)*3]))
            frame = np.vstack(hframes)
        else:
            frame = np.hstack(frames)
        frame_idx += 1
        # print(f"Display image: {i}.")
        current_time = time.time()
        current_fps = 1 / (current_time - last_time)
        last_time = current_time
        cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_title, frame)
        # for i in range(num_cameras):
        #     cameras[i][idx] = None
        key = cv2.waitKey(1000 // FPS // CAMERA_COUNT // 2)
        if key == 27 or check_count == CAMERA_COUNT:  # 按下ESC鍵退出
            stop.set()
            break
        elif key - ord('0') >= 0 and key - ord('0') < CAMERA_COUNT:
            idx = key - ord('0')
            if idx not in calibrate_queue:
                calibrate_queue.append(idx)
    cv2.destroyAllWindows()
    print("Stop display_images.")

def main(mode:Mode):
    stop = Event()
    # 創建相片的queue
    image_queue = Manager().Queue()
    cameras_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]
    display_queue = [Manager().Queue() for _ in range(CAMERA_COUNT)]
    is_found = Manager().list()
    calibrate_queue = None
    if mode == Mode.INTRINSIC:
        calibrate_queue = Manager().list()
        for i in range(CAMERA_COUNT):
            calibrate_queue.append(i)
    

    # 開啟相機並將相片放進queue的執行緒
    processes = []
    for i in range(CAMERA_COUNT):
        process = Process(target=process_camera, args=(i, cameras_queue[i], display_queue[i], is_found, stop, calibrate_queue, mode), name=f"process_camera_{i}")
        process.start()
        processes.append(process)

    display_images_p = Process(target=display_images, args=(display_queue, stop, calibrate_queue), name="display_images")
    display_images_p.start()
    processes.append(display_images_p)
    if TEST_CAMERA:
        capture_images_p = Process(target=test_capture_images, args=(cameras_queue,stop), name="capture_images")
    else:
        capture_images_p = Process(target=capture_images, args=(cameras_queue,stop), name="capture_images")
    capture_images_p.start()
    processes.append(capture_images_p)

    for process in processes:
        print("Join process: ", process.name)
        if process.is_alive():
            process.join()
        print("process: ", process.name, " is alive: ", process.is_alive())
        process.close()



if __name__ == "__main__":
    main(Mode.EXTRINSIC)

# TODO:
# 全部校正