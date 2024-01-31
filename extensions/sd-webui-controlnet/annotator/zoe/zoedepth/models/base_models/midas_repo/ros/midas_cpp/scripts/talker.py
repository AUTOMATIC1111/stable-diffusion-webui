#!/usr/bin/env python3


import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def talker():
    rospy.init_node('talker', anonymous=True)
    
    use_camera = rospy.get_param('~use_camera', False)
    input_video_file = rospy.get_param('~input_video_file','test.mp4')
    # rospy.loginfo(f"Talker - params: use_camera={use_camera}, input_video_file={input_video_file}")

    # rospy.loginfo("Talker: Trying to open a video stream")
    if use_camera == True:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_video_file)

    pub = rospy.Publisher('image_topic', Image, queue_size=1)
    rate = rospy.Rate(30) # 30hz
    bridge = CvBridge()

    while not rospy.is_shutdown():
        ret, cv_image = cap.read()
        if ret==False:
            print("Talker: Video is over")
            rospy.loginfo("Video is over")
            return

        try:
            image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("Talker: cv2image conversion failed: ", e)
            print(e)
            continue

        rospy.loginfo("Talker: Publishing frame")
        pub.publish(image)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
