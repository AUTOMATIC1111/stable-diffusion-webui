#!/usr/bin/env python3
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class video_show:

    def __init__(self):
        self.show_output = rospy.get_param('~show_output', True)
        self.save_output = rospy.get_param('~save_output', False)
        self.output_video_file = rospy.get_param('~output_video_file','result.mp4')
        # rospy.loginfo(f"Listener - params: show_output={self.show_output}, save_output={self.save_output}, output_video_file={self.output_video_file}")

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("midas_topic", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
            return

        if cv_image.size == 0:
            return

        rospy.loginfo("Listener: Received new frame")
        cv_image = cv_image.astype("uint8")

        if self.show_output==True:
            cv2.imshow("video_show", cv_image)
            cv2.waitKey(10)

        if self.save_output==True:
            if self.video_writer_init==False:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter(self.output_video_file, fourcc, 25, (cv_image.shape[1], cv_image.shape[0]))
            
            self.out.write(cv_image)



def main(args):
    rospy.init_node('listener', anonymous=True)
    ic = video_show()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)