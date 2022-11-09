import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch import actions

ROS_DISTRO = os.getenv('ROS_DISTRO')
if not ((ROS_DISTRO == "eloquent") or (ROS_DISTRO == "foxy") or (ROS_DISTRO == "humble")):
	print("ROS2 distribution " + ROS_DISTRO + " not recognised by launch file!")
	actions.Shutdown(reason="ROS2 distribution " + ROS_DISTRO + " not recognised by launch file!")
		
def generate_launch_description():
	
	package = 'detection_ros2_node'
	#name = 'fusion_node'
	executable = 'detection_ros2_node'
	namespace = ''

	config_filename = 'config.yaml'
	config_filename_default = 'config_default.yaml'

	config_dir = os.path.join(get_package_share_directory(package), 'config')
	config = os.path.join(config_dir, config_filename_default)
	if os.path.exists(os.path.join(config_dir, config_filename)):
		config = os.path.join(config_dir, config_filename)
		print("Load configuration from " + config_filename)
	else:
		print("Load default configuration")
	
	if ROS_DISTRO == "eloquent":
		return LaunchDescription([
			Node(
				package = package,
				node_namespace = namespace,
				node_executable = executable,
				name = 'gesture_det',
				parameters = [config],
				#parameter = {"debug": True},
				output = 'screen',
				arguments = ['--name', 'gesture_det'],
				#emulate_tty = True,
				#arguments = [('__log_level:=debug')]
			),
		])
	elif ROS_DISTRO == "foxy":
		return LaunchDescription([
			Node(
				package = package,
				namespace = namespace,
				executable = executable,
				name = 'gesture_det',
				parameters = [config],
				#parameter = {"debug": True},
				output = 'screen',
				arguments = ['--name', 'gesture_det'],
				#emulate_tty = True,
				#arguments = [('__log_level:=debug')]
			),
		])
	elif ROS_DISTRO == "humble":
		return LaunchDescription([
			Node(
				package = package,
				namespace = namespace,
				executable = executable,
				name = 'gesture_det',
				parameters = [config],
				#parameter = {"debug": True},
				output = 'screen',
				arguments = ['--name', 'gesture_det'],
				#emulate_tty = True,
				#arguments = [('__log_level:=debug')]
			),
		])
	else:
		return LaunchDescription()
