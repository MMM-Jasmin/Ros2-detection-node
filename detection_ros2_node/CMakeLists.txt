###########
## CMake ##
###########
cmake_minimum_required(VERSION 3.9)
message(STATUS "CMake version:\t" ${CMAKE_VERSION})
message(STATUS "CMAKE_ROOT:\t" ${CMAKE_ROOT})
cmake_policy(VERSION 3.9)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_WARN_DEPRECATED FALSE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE)


#############
## Project ##
#############
project(detection_ros2_node C CXX CUDA)
set(PROJECT_BINARY ${PROJECT_NAME})
set(PROJECT_LIBRARY ${PROJECT_NAME}_lib)

set(CMAKE_CXX_STANDARD 17)

# Build YoloLayer Plugin
#add_subdirectory(plugins)

find_package(TENSORRT PATHS cmake/modules REQUIRED)

file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c)
#file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/*.hpp ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/*.h ${PROJECT_SOURCE_DIR}/include/ )

#message(STATUS "SOURCE_FILES:\t" ${SOURCE_FILES})
#message(STATUS "HEADER_FILES:\t" ${HEADER_FILES})

add_executable(${PROJECT_BINARY} ${SOURCE_FILES} ${HEADER_FILES})
add_library(${PROJECT_LIBRARY} ${SOURCE_FILES} ${HEADER_FILES})

include_directories(SYSTEM
${TENSORRT_INCLUDE_DIRS}
${CUDAToolkit_INCLUDE_DIRS}
include/3rdParty/inipp
include/3rdParty/nvidia
)

include_directories(include/${PROJECT_NAME} plugins)

link_directories(${PROJECT_SOURCE_DIR}/libs)

##############
## Compiler ##
##############
# Default to C99
if(NOT CMAKE_C_STANDARD)
	set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set_target_properties(${PROJECT_BINARY}
	PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS YES
)

# Filesystem
target_link_libraries(${PROJECT_BINARY} stdc++fs ${TENSORRT_LIBRARIES} ${PROJECT_SOURCE_DIR}/libs/libYoloLayerPlugin.a ${CUDAToolkit_LIBRARY_DIR}/libcudart.so)
target_link_libraries(${PROJECT_LIBRARY} stdc++fs ${TENSORRT_LIBRARIES} ${PROJECT_SOURCE_DIR}/libs/libYoloLayerPlugin.a ${CUDAToolkit_LIBRARY_DIR}/libcudart.so)

##################
## Dependencies ##
##################
#### OpenCV ####
find_package(OpenCV REQUIRED)
message(STATUS "+==========[ OpenCV ]===========")
if (OpenCV_FOUND)
	message(STATUS "|  OpenCV version:\t" ${OpenCV_VERSION})
	#message(STATUS "|  OpenCV include:\t" ${OpenCV_INCLUDE_DIRS})
	#message(STATUS "|  OpenCV include:\t" ${OpenCV_LIBS})
	target_link_libraries(${PROJECT_BINARY} ${OpenCV_LIBS})
	target_link_libraries(${PROJECT_LIBRARY} ${OpenCV_LIBS})
else (OpenCV_FOUND)
	message(STATUS "|  OpenCV not found!")
endif (OpenCV_FOUND)
message(STATUS "+===================================")

#### ROS2 ####
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_interfaces REQUIRED)

# Preprocessor define for ros2 distribution eloquent elusor
if ($ENV{ROS_DISTRO} STREQUAL "eloquent")
	target_compile_definitions(${PROJECT_BINARY} PRIVATE ROS_ELOQUENT)
	target_compile_definitions(${PROJECT_LIBRARY} PRIVATE ROS_ELOQUENT)
endif ()
message(STATUS "+==========[ ROS2 ]===========")
message(STATUS "|  Distribution:\t" $ENV{ROS_DISTRO})
message(STATUS "+===================================")

###########
## Build ##
###########
target_include_directories(
	${PROJECT_BINARY} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
	${PROJECT_BINARY}
	"rclcpp"
	"sensor_msgs"
	"message_filters"
	"tf2_ros"
	"tf2_eigen"
	"ament_index_cpp"
	"tf2_geometry_msgs"
	"pcl_conversions"
	"image_transport"

	"camera_interfaces"
)

# library
ament_target_dependencies(
	${PROJECT_LIBRARY}
	"rclcpp"
	"sensor_msgs"
	"message_filters"
	"tf2_ros"
	"tf2_eigen"
	"ament_index_cpp"
	"tf2_geometry_msgs"
	"pcl_conversions"
	"image_transport"

	"camera_interfaces"
)

#############
## Install ##
#############
# Install launch, config and data files
install(
	DIRECTORY
	launch
	config
	DESTINATION share/${PROJECT_NAME}
)
# Install node executable
install(
	TARGETS ${PROJECT_BINARY}
	DESTINATION lib/${PROJECT_NAME}
)

# Install library
if ($ENV{ROS_DISTRO} STREQUAL "eloquent")
	ament_export_interfaces(${PROJECT_LIBRARY} HAS_LIBRARY_TARGET) # Eloquent
else()
	ament_export_targets(${PROJECT_LIBRARY} HAS_LIBRARY_TARGET) # Foxy
endif ()
install(
	DIRECTORY include/${PROJECT_NAME}
	DESTINATION include
)
install(
	TARGETS ${PROJECT_LIBRARY}
	EXPORT ${PROJECT_LIBRARY}
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	INCLUDES DESTINATION include
)

###################
## Documentation ##
###################
find_package(Doxygen)
if(DOXYGEN_FOUND)
	message(STATUS "Doxygen found")
	# Prepare doxygen configuration file
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
	# Add doc as target
	add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
endif(DOXYGEN_FOUND)

ament_package()
