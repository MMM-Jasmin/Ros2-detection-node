#include "detection_node.hpp"

#include "YoloTRT.h"
#include "SORT.h"

const double ONE_SECOND            = 1000.0; // One second in milliseconds

/**
 * @brief Contructor.
 */
DetectionNode::DetectionNode(const std::string &name) : Node(name, rclcpp::NodeOptions().use_intra_process_comms(false)) 
{
	this->declare_parameter("rotation", 0);
	this->declare_parameter("debug", false);
	this->declare_parameter("topic", "");
	this->declare_parameter("print_detections", true);
	this->declare_parameter("print_fps", true);
	this->declare_parameter("det_topic", "test/det");
	this->declare_parameter("fps_topic", "test/fps");
	this->declare_parameter("max_fps", 30.0f);
	this->declare_parameter("qos_sensor_data", true);
	this->declare_parameter("qos_history_depth", 10);
	 
    this->declare_parameter("DLA_CORE", 0);
    this->declare_parameter("USE_FP16", true);
    this->declare_parameter("ONNX_FILE", "");
    this->declare_parameter("CONFIG_FILE", "");
    this->declare_parameter("ENGINE_FILE" , "");
    this->declare_parameter("CLASS_FILE", "");
    this->declare_parameter("DETECT_STR", "");
    this->declare_parameter("AMOUNT_STR", "");
    this->declare_parameter("FPS_STR", "");
    this->declare_parameter("YOLO_VERSION", 4);
    this->declare_parameter("YOLO_TINY", true);
    this->declare_parameter("YOLO_THRESHOLD", 0.3);	

	callback_handle_ = this->add_on_set_parameters_callback(std::bind(&DetectionNode::parametersCallback, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult DetectionNode::parametersCallback(const std::vector<rclcpp::Parameter> &parameters)
{

	for (const auto &param: parameters){
		if (param.get_name() == "max_fps")
			m_maxFPS = param.as_double();
	}

	rcl_interfaces::msg::SetParametersResult result;
	result.successful = true;
    result.reason = "success";
	return result;
}

/**
 * @brief Initialize image node.
 */
void DetectionNode::init() {


	int YOLO_VERSION, DLA_CORE, qos_history_depth;
	bool USE_FP16, YOLO_TINY, qos_sensor_data;
	float YOLO_THRESHOLD;
	std::string ONNX_FILE, CONFIG_FILE, ENGINE_FILE, CLASS_FILE, ros_topic, det_topic, fps_topic;

	std::cout << "-- get ros config variables --" << std::endl;

	// needed only for init
	// get ros configuration
	this->get_parameter("topic", ros_topic);
	this->get_parameter("det_topic", det_topic);
	this->get_parameter("fps_topic", fps_topic);	
	
	// get Yolo configuration
 	this->get_parameter("DLA_CORE", DLA_CORE);
    this->get_parameter("USE_FP16", USE_FP16);
    this->get_parameter("ONNX_FILE", ONNX_FILE);
    this->get_parameter("CONFIG_FILE", CONFIG_FILE);
    this->get_parameter("ENGINE_FILE" , ENGINE_FILE);
    this->get_parameter("CLASS_FILE", CLASS_FILE);
    this->get_parameter("YOLO_VERSION", YOLO_VERSION);
    this->get_parameter("YOLO_TINY", YOLO_TINY);
    this->get_parameter("YOLO_THRESHOLD", YOLO_THRESHOLD);	

	// some things needs to be member
	this->get_parameter("max_fps", m_maxFPS);
	this->get_parameter("DETECT_STR", m_DETECT_STR);
    this->get_parameter("AMOUNT_STR", m_AMOUNT_STR);
    this->get_parameter("FPS_STR", m_FPS_STR);
	this->get_parameter("rotation", m_image_rotation);
	this->get_parameter("print_detections", m_print_detections);
	this->get_parameter("print_fps", m_print_fps);
	this->get_parameter("qos_sensor_data", qos_sensor_data);
	this->get_parameter("qos_history_depth", qos_history_depth);

	YoloType yoloType = YoloType::NON;

		if (YOLO_VERSION == 3)
			yoloType |= YoloType::YOLO_V3;
		else if (YOLO_VERSION == 4)
			yoloType |= YoloType::YOLO_V4;
		else
			std::cerr << "Invalid version (" << YOLO_VERSION << ") specified in YOLO_VERSION" << std::endl;

		if (YOLO_TINY == true)
				yoloType |= YoloType::TINY;

	std::cout << "-- init tensorrt --" << std::endl;

	// Set TensorRT log level
	TrtLog::gLogger.setReportableSeverity(TrtLog::Severity::kWARNING);

	m_pYolo = new YoloTRT(ONNX_FILE, CONFIG_FILE, ENGINE_FILE, CLASS_FILE, DLA_CORE, USE_FP16, true, YOLO_THRESHOLD, yoloType);

	m_pSortTrackers = new SORT[m_pYolo->GetClassCount()];
	////// Initialize SORT tracker for each class
	for (std::size_t i = 0; i < m_pYolo->GetClassCount(); i++)
		m_pSortTrackers[i] = SORT(30, 5);

	m_lastTrackings.clear();

	m_elapsedTime = 0;
	m_timer.Start();
	m_last_str = "";

	std::cout << "-- subscribe to : " << ros_topic <<  " --" << std::endl;

	if(qos_sensor_data){
		std::cout << "using ROS2 qos_sensor_data" << std::endl;
		m_qos_profile = rclcpp::SensorDataQoS();
	}

	m_qos_profile = m_qos_profile.keep_last(qos_history_depth);
	//m_qos_profile = m_qos_profile.lifespan(std::chrono::milliseconds(500));
	m_qos_profile = m_qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
	m_qos_profile = m_qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
	

	m_qos_profile_sysdef = m_qos_profile_sysdef.keep_last(qos_history_depth);
	//m_qos_profile_sysdef = m_qos_profile_sysdef.lifespan(std::chrono::milliseconds(500));
	m_qos_profile_sysdef = m_qos_profile_sysdef.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
	m_qos_profile_sysdef = m_qos_profile_sysdef.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
	

	
	//m_qos_profile_sysdef = m_qos_profile_sysdef.keep_last(qos_history_depth);
	//m_qos_profile = m_qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
	//m_qos_profile = m_qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
	//m_qos_profile = m_qos_profile.deadline(std::chrono::nanoseconds(static_cast<int>(1e9 / 30)));


	m_image_small_subscription = this->create_subscription<sensor_msgs::msg::Image>( ros_topic, m_qos_profile, std::bind(&DetectionNode::imageSmallCallback, this, std::placeholders::_1));
	//cv::namedWindow(m_window_name_image_small, cv::WINDOW_AUTOSIZE);

	std::cout << "-- create topics for publishing --" << std::endl;

	m_detection_publisher   = this->create_publisher<std_msgs::msg::String>(det_topic, m_qos_profile_sysdef);
	m_fps_publisher    		= this->create_publisher<std_msgs::msg::String>(fps_topic, m_qos_profile_sysdef);

	std::cout << "+==========[ init done ]==========+" << std::endl;

}


/**
 * @brief Callback function for reveived image message.
 * @param img_msg Received image message
 */
void DetectionNode::imageSmallCallback(sensor_msgs::msg::Image::SharedPtr img_msg) {

	cv::Size image_size(static_cast<int>(img_msg->width), static_cast<int>(img_msg->height));
	cv::Mat color_image(image_size, CV_8UC3, (void *)img_msg->data.data(), cv::Mat::AUTO_STEP);
	
	if (m_image_rotation == 90)
		cv::rotate(color_image, color_image, cv::ROTATE_90_CLOCKWISE);
	else if (m_image_rotation == 180)
		cv::rotate(color_image, color_image, cv::ROTATE_180);
	else if (m_image_rotation == 270)
		cv::rotate(color_image, color_image, cv::ROTATE_90_COUNTERCLOCKWISE); 

	m_frame = color_image;

	ProcessNextFrame();
	ProcessDetections();
	
	m_frameCnt++;
	CheckFPS(&m_frameCnt);

	//v::setWindowTitle(m_window_name_image_small, std::to_string(m_loop_duration_image_small));
	//cv::cvtColor(color_image, color_image, cv::COLOR_RGB2BGR);
	//imshow(m_window_name_image_small, color_image);

	//if (!(cv::waitKey(1) < 0 && cv::getWindowProperty(m_window_name_image_small, cv::WND_PROP_AUTOSIZE) >= 0))
	//	rclcpp::shutdown();


	//m_loop_duration_image_small = (hires_clock::now() - m_callback_time_image_small).count() / 1e6;
	//m_callback_time_image_small = hires_clock::now();
}

/**
 * @brief Declare DepthFusion ros node parameters.
 */
void DetectionNode::declareNodeParameters()
{

}

void DetectionNode::ProcessDetections( )
{
	bool changed                        = false;
	//const ::YoloResults& results = m_yoloResults;

	std::map<uint32_t, TrackingObjects> trackingDets;

	for (const YoloResult& r : m_yoloResults)
	{
		trackingDets.try_emplace(r.ClassID(), TrackingObjects());
		trackingDets[r.ClassID()].push_back({ { r.x, r.y, r.w, r.h }, static_cast<uint32_t>(std::round(r.Conf() * 100)), m_pYolo->ClassName(r.ClassID()) });
	}

	TrackingObjects trackers;
	TrackingObjects dets;

	for (std::size_t i = 0; i < m_pYolo->GetClassCount(); i++)
	{
		if (trackingDets.count(i))
			dets = trackingDets[i];
		else
			dets = TrackingObjects();
			TrackingObjects t = m_pSortTrackers[i].Update(dets);
		trackers.insert(std::end(trackers), std::begin(t), std::end(t));
	}

	if (trackers.size() != m_lastTrackings.size())
		changed = true;
	else
	{
		for (const auto& [idx, obj] : enumerate(trackers))
		{
			if (m_lastTrackings[idx] != obj)
			{
				changed = true;
				break;
			}
		}
	}
		if (changed)
			printDetections(trackers); 
}

void DetectionNode::ProcessNextFrame()
{
	if (!m_frame.empty())
		m_yoloResults = m_pYolo->Infer(m_frame);
} 

BBox DetectionNode::toCenter(const BBox& bBox)
{
	// x_y = center
	float h = bBox.height;
	float w = bBox.width;
	float x = bBox.x + (w / 2);
	float y = bBox.y + (h / 2);
	return BBox(x, y, w, h);
}

void DetectionNode::printDetections(const TrackingObjects& trackers)
{
	std::stringstream str("");
	str << string_format("{\"%s\": [", m_DETECT_STR.c_str());

	for (const auto& [i, t] : enumerate(trackers))
	{
		BBox centerBox = toCenter(t.bBox);
		str << string_format("{\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.3f,%.3f], \"w_h\": [%.3f,%.3f]}", t.trackingID, t.name.c_str(), roundf(centerBox.x*1000.0f)/1000.0f , roundf(centerBox.y*1000.0f)/1000.0f, roundf(centerBox.width*1000.0f)/1000.0f, roundf(centerBox.height*1000.0f)/1000.0f);
		// Prevent a trailing ',' for the last element
		if (i + 1 < trackers.size()) str << ", ";
	}

	m_lastTrackings = trackers;

	str << string_format("], \"%s\": %llu }", m_AMOUNT_STR.c_str(), m_lastTrackings.size());

	auto message = std_msgs::msg::String();
	message.data = str.str();
	try{
		m_detection_publisher->publish(message);
	}
	catch (...) {
		RCLCPP_INFO(this->get_logger(), "hmm publishing dets has failed!! ");
	}

	if (m_print_detections)
		RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());

	std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int32_t>(std::round(5))));
	
}

void DetectionNode::CheckFPS(uint64_t* pFrameCnt)
	{
		m_timer.Stop();

		double minFrameTime = 1000.0 / m_maxFPS;
		double itrTime      = m_timer.GetElapsedTimeInMilliSec();
		double fps;

		m_elapsedTime += itrTime;

		fps = 1000 / (m_elapsedTime / (*pFrameCnt));

		if (m_elapsedTime >= ONE_SECOND)
		{
			PrintFPS(fps, itrTime);

			*pFrameCnt    = 0;
			m_elapsedTime = 0;
		}

		m_timer.Start();
	}

void DetectionNode::PrintFPS(const float fps, const float itrTime)
{
		
	std::stringstream str("");

	if (fps == 0.0f)
			str << string_format("{\"%s\": 0.0}", m_FPS_STR.c_str());
	else
		str << string_format("{\"%s\": %.2f, \"lastCurrMSec\": %.2f, \"maxFPS\": %.2f, \"%s\": %llu }", m_FPS_STR.c_str(), fps, itrTime, m_maxFPS, m_AMOUNT_STR.c_str(), m_lastTrackings.size());

	auto message = std_msgs::msg::String();
	message.data = str.str();
	
	try{
		m_fps_publisher->publish(message);
	}
  	catch (...) {
    	RCLCPP_INFO(this->get_logger(), "m_fps_publisher: hmm publishing dets has failed!! ");
  	}

		
	if (m_print_fps)
		RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());

}