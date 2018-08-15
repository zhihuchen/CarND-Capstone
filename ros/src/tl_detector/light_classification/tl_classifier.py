import rospy
from styx_msgs.msg import TrafficLight
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer

DEBUG = True

class TLClassifier(object):
    def __init__(self):

        PATH_TO_GRAPH = r'light_classification/model/ssd_sim/frozen_inference_graph.pb'

        self.graph = tf.Graph()
        self.threshold = 0.525

        with self.graph.as_default():
            object_detection_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                object_detection_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(object_detection_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            image_expand = np.expand_dims(image, axis=0)
            start_time = timer()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expand})
            cycle_time_classification = timer() - start_time

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if DEBUG:
            rospy.loginfo("Classification Score: %s" % scores[0])
            rospy.loginfo("Classification Class: %s" % classes[0])
            rospy.loginfo("Classification Time : %s" % cycle_time_classification)

        if scores[0] > self.threshold:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
