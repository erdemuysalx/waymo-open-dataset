# Import necessary modules
import os
import json
import datetime
import argparse
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm

# Import Waymo Open Dataset modules
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract metadata from dataset')
    parser.add_argument('--input_dir', dest='input', type=str, required=True,
                        help='Input dataset directory')
    parser.add_argument('--output_file', dest='output', type=str, required=True,
                        help='JSON file for storing the output')
    args = parser.parse_args()

    # Define paths
    input_dir = args.input  # '/media/rauysal/My Book/waymo/v1.4/individual_files/'
    output_fname = args.output  # output.json
    # Define data directories
    data_dirs = ['training/', 'validation/', 'testing/', 'domain_adaptation/', 'testing_3d_camera_only_detection/']
    # Storage for file paths
    file_paths = []

    # Total number of frames
    frame_count = 0
    # Total number of objects detected in images
    total_obj_2d_count = 0
    total_obj_3d_count = 0
    # Total number of useful objects detected in images
    # Usefulness refers to being pedestrian or cyclist
    useful_obj_2d_count = 0
    useful_obj_3d_count = 0
    # Storage for detected objects' labels, in order to find unique counts
    obj_2d_labels = []
    obj_3d_labels = []
    # Storage for detected useful objects' labels, in order to find unique counts
    useful_obj_2d_labels = []
    useful_obj_3d_labels = []
    # Class counts
    vehicle_2d_count = 0
    pedestrian_2d_count = 0
    cyclist_2d_count = 0
    sign_2d_count = 0
    vehicle_3d_count = 0
    pedestrian_3d_count = 0
    cyclist_3d_count = 0
    sign_3d_count = 0
    # Storage for class labels, in order to find unique class counts
    vehicle_2d_labels = []
    pedestrian_2d_labels = []
    cyclist_2d_labels = []
    sign_2d_labels = []
    vehicle_3d_labels = []
    pedestrian_3d_labels = []
    cyclist_3d_labels = []
    sign_3d_labels = []
    # Total number of 2D to 3D correspondence
    correspondence_2d3d_count = 0
    correspondence_2d3d_labels = []
    # FPS in Hz
    frame_rate = 10
    ##
    frames_per_track_id = defaultdict(int)
    seconds_per_track_id = None
    avg_tracking_time = 0

    # Create files that are not exist
    if not os.path.isfile(output_fname):
        with open(output_fname, 'w') as f:
            f.close()

    # Loop through data directories
    for dir in data_dirs:
        print(f"[INFO] Extracting {dir[0:-1]} data")
        split_count = 0
        for root, dirs, files in os.walk(os.path.join(input_dir, dir)):
            for file in files:
                if file.endswith('.tfrecord'):
                    # Get all TFRecord files
                    path = str(root) + '/' + str(file)
                    file_paths.append(path)
                    split_count += 1

        print(f"[INFO] Extracting {dir[0:-1]} data is completed")
        print(f"[INFO] There are {split_count} TFRecord file for {dir[0:-1]}")
        print("============================================================================================================")
    print(f"[INFO] There are {len(file_paths)} TFRecord file in total")
    print()

    # Loop through TFRecord files
    for frame_path in tqdm(file_paths):
        # Unpack dataset frin TFRecord file
        dataset = tf.data.TFRecordDataset(frame_path, compression_type='')
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # Group labels of detected objects on a single frame
            # e.g. vehicle, pedestrian etc.
            #labels = keypoint_data.group_object_labels(frame)
            # Increase frame count by one
            frame_count += 1

            # Loop through objects detected via camera
            for camera_labels in frame.camera_labels:
                for label in camera_labels.labels:
                    # total_obj_2d_count += 1
                    # if label.id not in obj_2d_labels:
                    #     obj_2d_labels.append(label.id)

                    # if label.type == label_pb2.Label.TYPE_PEDESTRIAN or \
                    #     label.type == label_pb2.Label.TYPE_CYCLIST:
                    #     useful_obj_2d_count += 1
                    #     if label.id not in useful_obj_2d_labels:
                    #         useful_obj_2d_labels.append(label.id)

                    # If the object type is vehicle
                    if label.type == label_pb2.Label.TYPE_VEHICLE:
                        # Increase vehicle count by one
                        vehicle_2d_count += 1
                        if label.id not in vehicle_2d_labels:
                            vehicle_2d_labels.append(label.id)

                    # If the object type is pedestrian
                    elif label.type == label_pb2.Label.TYPE_PEDESTRIAN:
                        # Increase pedestrian count by one
                        pedestrian_2d_count += 1
                        if label.id not in pedestrian_2d_labels:
                            pedestrian_2d_labels.append(label.id)

                    # If the object type is cyclist
                    elif label.type == label_pb2.Label.TYPE_CYCLIST:
                        # Increase cyclist count by one
                        cyclist_2d_count += 1
                        if label.id not in cyclist_2d_labels:
                            cyclist_2d_labels.append(label.id)

                    # If the object type is sign
                    elif label.type == label_pb2.Label.TYPE_SIGN:
                        # Increase sign count by one
                        sign_2d_count += 1
                        if label.id not in sign_2d_labels:
                            sign_2d_labels.append(label.id)

                    if label.association:
                        # This operation saves the object's id into the container
                        # if the object label has an association feature, 2D to 3D correspondance
                        # and starts counting the frame number of the object label
                        #
                        # Length of this container is equivalent to total number of
                        # objects which has 2D to 3D correspondance
                        # This number is also equal to count of 2D unique objects
                        frames_per_track_id[label.id] += 1

                        # if label.id not in correspondence_2d3d_labels:
                        #     correspondence_2d3d_labels.append(label.id)

            # Loop through objects detected via lidar
            for laser_label in frame.laser_labels:
                # total_obj_3d_count += 1
                # if laser_label.id not in obj_3d_labels:
                #     obj_3d_labels.append(laser_label.id)

                # If the object label has a laser annotation
                # if laser_label.type == label_pb2.Label.TYPE_PEDESTRIAN or \
                #     laser_label.type == label_pb2.Label.TYPE_CYCLIST:
                #     # Increase useful 3D objects count by one
                #     useful_obj_3d_count += 1
                #     if laser_label.id not in useful_obj_3d_labels:
                #         useful_obj_3d_labels.append(laser_label.id)

                # If the object type is vehicle
                if laser_label.type == label_pb2.Label.TYPE_VEHICLE:
                    # Increase vehicle count by one
                    vehicle_3d_count += 1
                    if laser_label.id not in vehicle_3d_labels:
                        vehicle_3d_labels.append(laser_label.id)

                # If the object type is pedestrian
                elif laser_label.type == label_pb2.Label.TYPE_PEDESTRIAN:
                    # Increase pedestrian count by one
                    pedestrian_3d_count += 1
                    if laser_label.id not in pedestrian_3d_labels:
                        pedestrian_3d_labels.append(laser_label.id)

                # If the object type is cyclist
                elif laser_label.type == label_pb2.Label.TYPE_CYCLIST:
                    # Increase cyclist count by one
                    cyclist_3d_count += 1
                    if laser_label.id not in cyclist_3d_labels:
                        cyclist_3d_labels.append(laser_label.id)

                # If the object type is sign
                elif laser_label.type == label_pb2.Label.TYPE_SIGN:
                    # Increase sign count by one
                    sign_3d_count += 1
                    if laser_label.id not in sign_3d_labels:
                        sign_3d_labels.append(laser_label.id)

            try:
                seconds_per_track_id = defaultdict(int, {k: v / frame_rate for k, v in frames_per_track_id.items()})
                avg_tracking_time = sum(seconds_per_track_id.values()) / len(seconds_per_track_id)
            except ZeroDivisionError as err:
                print("Division by zero!:", err)

            print("============================================================================================================")
            print(f"[INFO] Loaded {frame_count} frames in total")
            # print(f"[INFO] Loaded {vehicle_2d_count + pedestrian_2d_count + cyclist_2d_count + sign_2d_count} 2D objects in total")
            # print(f"[INFO] Loaded {vehicle_3d_count + pedestrian_3d_count + cyclist_3d_count + sign_3d_count} 3D objects in total")
            # print(f"[INFO] {len(vehicle_2d_labels) + len(pedestrian_2d_labels) + len(cyclist_2d_labels) + len(sign_2d_labels)} 2D unique objects")
            # print(f"[INFO] {len(vehicle_3d_labels) + len(pedestrian_3d_labels) + len(cyclist_3d_labels) + len(sign_3d_labels)} 3D unique objects")
            # print(f"[INFO] {pedestrian_2d_count + cyclist_2d_count} 2D useful objects")
            # print(f"[INFO] {pedestrian_3d_count + cyclist_3d_count} 3D useful objects")
            # print(f"[INFO] {len(pedestrian_2d_labels) + len(cyclist_2d_labels)} 2D unique useful objects")
            # print(f"[INFO] {len(pedestrian_3d_labels) + len(cyclist_3d_labels)} 3D unique usefull objects")
            # print()
            # print(f"[INFO] {vehicle_2d_count} 2D vehicle objects")
            # print(f"[INFO] {pedestrian_2d_count} 2D pedestrian objects")
            # print(f"[INFO] {cyclist_2d_count} 2D cyclist objects")
            # print(f"[INFO] {sign_2d_count} 2D sign objects")
            # print(f"[INFO] {len(vehicle_2d_labels)} 2D unique vehicle objects")
            # print(f"[INFO] {len(pedestrian_2d_labels)} 2D unique pedestrian objects")
            # print(f"[INFO] {len(cyclist_2d_labels)} 2D unique cyclist objects")
            # print(f"[INFO] {len(sign_2d_labels)} 2D unique sign objects")
            # print()
            # print(f"[INFO] {vehicle_3d_count} 3D vehicle objects")
            # print(f"[INFO] {pedestrian_3d_count} 3D pedestrian objects")
            # print(f"[INFO] {cyclist_3d_count} 3D cyclist objects")
            # print(f"[INFO] {sign_3d_count} 3D sign objects")
            # print(f"[INFO] {len(vehicle_3d_labels)} 3D unique vehicle objects")
            # print(f"[INFO] {len(pedestrian_3d_labels)} 3D unique pedestrian objects")
            # print(f"[INFO] {len(cyclist_3d_labels)} 3D unique cyclist objects")
            # print(f"[INFO] {len(sign_3d_labels)} 3D unique sign objects")
            # print()
            #print(f"[INFO] {correspondence_2d3d_count} 2D to 3D correspondence")
            #print(f"[INFO] {len(correspondence_2d3d_labels)} len labels 2D to 3D correspondence")
            print(f"[INFO] {len(frames_per_track_id)} len frames per track id 2D to 3D correspondence")
            print(f"[INFO] Duration: {str(datetime.timedelta(seconds=frame_count / frame_rate))}")
            print(f"[INFO] Average tracking time: {avg_tracking_time}")

            metadata = {
                'frame_count': frame_count,

                'obj_count': {'2D': vehicle_2d_count + pedestrian_2d_count + cyclist_2d_count + sign_2d_count,  # obj_2d_count
                              '3D': vehicle_3d_count + pedestrian_3d_count + cyclist_3d_count + sign_3d_count},  # obj_3d_count

                'unique_obj_count': {'2D': len(vehicle_2d_labels) + len(pedestrian_2d_labels) + len(cyclist_2d_labels) + len(sign_2d_labels),  # len(obj_2d_labels)
                                     '3D': len(vehicle_3d_labels) + len(pedestrian_3d_labels) + len(cyclist_3d_labels) + len(sign_3d_labels)},  # len(obj_3d_labels)

                'useful_obj_count': {'2D': pedestrian_2d_count + cyclist_2d_count,  # useful_obj_2d_count
                                     '3D': pedestrian_3d_count + cyclist_3d_count},  # useful_obj_3d_count

                'unique_useful_obj_count': {'2D': len(pedestrian_2d_labels) + len(cyclist_2d_labels),  # len(useful_obj_2d_labels)
                                            '3D': len(pedestrian_3d_labels) + len(cyclist_3d_labels)},  # len(useful_obj_3d_labels)


                'class_count': {'2D': {'vehicle': vehicle_2d_count,
                                       'pedestrian': pedestrian_2d_count,
                                       'cyclist': cyclist_2d_count,
                                       'sign': sign_2d_count},
                                '3D': {'vehicle': vehicle_3d_count,
                                       'pedestrian': pedestrian_3d_count,
                                       'cyclist': cyclist_3d_count,
                                       'sign': sign_3d_count}},

                'unique_class_count': {'2D': {'vehicle': len(vehicle_2d_labels),
                                              'pedestrian': len(pedestrian_2d_labels),
                                              'cyclist': len(cyclist_2d_labels),
                                              'sign': len(sign_2d_labels)},
                                       '3D': {'vehicle': len(vehicle_3d_labels),
                                              'pedestrian': len(pedestrian_3d_labels),
                                              'cyclist': len(cyclist_3d_labels),
                                              'sign': len(sign_3d_labels)}},

                'correspondence_2d3d_count': len(frames_per_track_id),
                'avg_tracking_time': avg_tracking_time,
                'duration': (str(datetime.timedelta(seconds=frame_count / frame_rate))),
            }

            with open(output_fname, 'w') as f:
                json.dump(metadata, f)
