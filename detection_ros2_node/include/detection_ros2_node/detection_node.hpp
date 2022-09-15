#pragma once
// SYSTEM
#include <chrono>
#include <iostream>
// ROS
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "std_msgs/msg/string.hpp"
// OPENCV
#include <opencv2/opencv.hpp>

// PROJECT
#include "camera_interfaces/msg/depth_frameset.hpp"

#include "Types.h"
#include "Timer.h"


/**
 * @brief Image viewer node class for receiving and visualizing fused image.
 */
class DetectionNode : public rclcpp::Node
{
	typedef std::chrono::high_resolution_clock::time_point time_point;
	typedef std::chrono::high_resolution_clock hires_clock;

public:
	DetectionNode(const std::string &name);
	void init();

private:

	uint64_t m_frameCnt = 0;

	float m_maxFPS;
	int m_image_rotation;
	bool m_print_detections, m_print_fps;
	std::string m_DETECT_STR, m_AMOUNT_STR, m_FPS_STR, m_last_str;

	Timer m_timer;        // Timer used to measure the time required for one iteration
	double m_elapsedTime; // Sum of the elapsed time, used to check if one second has passed
	
	//  ========= Yolo Node =========
	TrackingObjects m_lastTrackings; // Vector containing the last tracked objects
	class SORT* m_pSortTrackers;           // Pointer to n-sort trackers (n = number of classes)

	class YoloTRT* m_pYolo;                          // Pointer to a TensorRT Yolo object used to process the input image
	YoloResults m_yoloResults; // Buffer for the yolo results

	cv::Mat m_frame; // Buffer for the input frame

	void declareNodeParameters();

	std::string m_window_name_image_small	= "Image_small_Frame";

	time_point m_callback_time = hires_clock::now();
	time_point m_callback_time_image_small = hires_clock::now();
	time_point m_callback_time_depth = hires_clock::now();

	double m_loop_duration = 0.0;
	double m_loop_duration_image_small = 0.0;
	double m_loop_duration_depth = 0.0;


	rclcpp::QoS m_qos_profile = rclcpp::SystemDefaultsQoS();
	rclcpp::QoS m_qos_profile_sysdef = rclcpp::SystemDefaultsQoS();
	
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_detection_publisher 	= nullptr;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_fps_publisher 	=	 nullptr;

	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_small_subscription;

	OnSetParametersCallbackHandle::SharedPtr callback_handle_;

	void imageSmallCallback(sensor_msgs::msg::Image::SharedPtr img_msg);

	rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);
	void ProcessDetections();
	void ProcessNextFrame();
	BBox toCenter(const BBox& bBox);
	void printDetections(const TrackingObjects& trackers);
	void CheckFPS(uint64_t* pFrameCnt);
	void PrintFPS(const float fps, const float itrTime);
};
