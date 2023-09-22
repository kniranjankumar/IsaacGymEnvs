#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/Float64.h"
#include "sensor_msgs/JointState.h"

#include <nav_msgs/Path.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#define updateRate 100 // Downsample rate 20, so 100Hz update rate is real time.
#define pi 3.1415926
#define traj_length 100 // Number of columns in state matrix.
#define num_state 20 // Number of rows in state matrix.

sensor_msgs::JointState createcp(Eigen::VectorXd state){
  std_msgs::Header head;
  	sensor_msgs::JointState cassie_joint_state;
  	head.stamp = ros::Time::now();
  	cassie_joint_state.header = head;
    //calc point
  	cassie_joint_state.name.push_back("pelvis_x");
  	cassie_joint_state.name.push_back("pelvis_y");
    cassie_joint_state.name.push_back("pelvis_z");
    cassie_joint_state.name.push_back("pelvis_yaw");

    cassie_joint_state.name.push_back("hip_abduction_left");
    cassie_joint_state.name.push_back("hip_rotation_left");
    cassie_joint_state.name.push_back("hip_flexion_left");
    cassie_joint_state.name.push_back("knee_joint_left");
    // cassie_joint_state.name.push_back("knee_to_shin_left");
    cassie_joint_state.name.push_back("shin_to_tarsus_left");
    cassie_joint_state.name.push_back("toe_pitch_joint_left");
    cassie_joint_state.name.push_back("toe_roll_joint_left");

    cassie_joint_state.name.push_back("shoulder_roll_joint_left");
    cassie_joint_state.name.push_back("shoulder_pitch_joint_left");
    cassie_joint_state.name.push_back("shoulder_yaw_joint_left");
    cassie_joint_state.name.push_back("elbow_joint_left");

    cassie_joint_state.name.push_back("hip_abduction_right");
    cassie_joint_state.name.push_back("hip_rotation_right");
    cassie_joint_state.name.push_back("hip_flexion_right");
    cassie_joint_state.name.push_back("knee_joint_right");
    // cassie_joint_state.name.push_back("knee_to_shin_right");
    cassie_joint_state.name.push_back("shin_to_tarsus_right");
    cassie_joint_state.name.push_back("toe_pitch_joint_right");
    cassie_joint_state.name.push_back("toe_roll_joint_right");

    cassie_joint_state.name.push_back("shoulder_roll_joint_right");
    cassie_joint_state.name.push_back("shoulder_pitch_joint_right");
    cassie_joint_state.name.push_back("shoulder_yaw_joint_right");
    cassie_joint_state.name.push_back("elbow_joint_right");
    // Two extra joint to visualize rod collision.
    // cassie_joint_state.name.push_back("hip_flexion_rod_left");
    // cassie_joint_state.name.push_back("hip_flexion_rod_right");

  	cassie_joint_state.position.push_back(state(0));
		cassie_joint_state.position.push_back(state(1));
		cassie_joint_state.position.push_back(state(2));
    cassie_joint_state.position.push_back(state(3));

    cassie_joint_state.position.push_back(-state(6)+0.37);
  	cassie_joint_state.position.push_back(-state(7));
    cassie_joint_state.position.push_back(-state(8)+0.76);
  	cassie_joint_state.position.push_back(state(9)+1.3);
    cassie_joint_state.position.push_back(state(11)-1.24);
    cassie_joint_state.position.push_back(state(12)+1.3);
    cassie_joint_state.position.push_back(0);

    cassie_joint_state.position.push_back(0);
    cassie_joint_state.position.push_back(1.2);
    cassie_joint_state.position.push_back(0);
    cassie_joint_state.position.push_back(-0.7);

  	cassie_joint_state.position.push_back(-state(13)-0.37);
    cassie_joint_state.position.push_back(-state(14));
  	cassie_joint_state.position.push_back(state(15)-0.76);
    cassie_joint_state.position.push_back(-state(16)-1.3);
    cassie_joint_state.position.push_back(-state(18)+1.24);
  	cassie_joint_state.position.push_back(-state(19)-1.3);
    cassie_joint_state.position.push_back(0);

    cassie_joint_state.position.push_back(0);
    cassie_joint_state.position.push_back(-1.2);
    cassie_joint_state.position.push_back(0);
    cassie_joint_state.position.push_back(0.7);

    // cassie_joint_state.position.push_back(state(8)+state(9)+0.13);
  	// cassie_joint_state.position.push_back(state(15)+state(16)+0.13);   
    return cassie_joint_state;
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "JointsPublisher");
	ros::NodeHandle n;
	ros::Publisher pub = n.advertise<sensor_msgs::JointState>("/digit/joint_states", 1000);
  ros::Publisher foot_pub1 = n.advertise<visualization_msgs::Marker>( "visualization_marker1", 0 );
  ros::Publisher foot_pub2 = n.advertise<visualization_msgs::Marker>( "visualization_marker2", 0 );
  ros::Publisher foot_pub3 = n.advertise<visualization_msgs::Marker>( "visualization_marker3", 0 );
  ros::Rate loop_rate(updateRate);

  std::ifstream infile("full_trajectory.txt"); // Need to run this code under the same folder.
  int row = 0;
  double height_offset = 0;

  while(ros::ok()) {
    Eigen::Matrix<double, num_state, 1> traj;
    std::string line;
    if (!std::getline(infile, line)) {
      std::cout << "Finish read.\n";
      break;
    }
    
    // std::vector<std::vector<double>> buff(num_state, std::vector<double>(traj_length, 0));
    std::istringstream ss(line);
    std::string number;
    int col = 0;
    while(std::getline(ss, number, ',')) {
        // std::cout << std::stod(number) << '\n';
        traj(col++,0) = std::stod(number);
    }
    // std::cout << col << '\n';

    // if (row == 0)
    //   height_offset = traj(2,0) - 0.9;
    row++;
    traj(2,0) -= height_offset;

    // Set robot joint state.
    sensor_msgs::JointState cassie_joint_state = createcp(traj);
    pub.publish(cassie_joint_state);
    
    // Publish Swing foot target.
    // visualization_msgs::Marker marker_d;
    // marker_d.header.frame_id = "cassie/fixed_link";
    // marker_d.header.stamp = ros::Time();
    // marker_d.ns = "my_namespace";
    // marker_d.id = 0;
    // marker_d.type = visualization_msgs::Marker::SPHERE;
    // marker_d.action = visualization_msgs::Marker::ADD;
    // marker_d.pose.position.x = traj(40);
    // marker_d.pose.position.y = traj(41);
    // marker_d.pose.position.z = traj(2)-0.9;
    // marker_d.pose.orientation.x = 0.0;
    // marker_d.pose.orientation.y = 0.0;
    // marker_d.pose.orientation.z = 0.0;
    // marker_d.pose.orientation.w = 1.0;
    // marker_d.scale.x = 0.05;
    // marker_d.scale.y = 0.05;
    // marker_d.scale.z = 0.05;
    // marker_d.color.a = 1.0; // Don't forget to set the alpha!
    // marker_d.color.r = 0.0;
    // marker_d.color.g = 1.0;
    // marker_d.color.b = 0.0;
    // foot_pub1.publish( marker_d);

    // Publish left foot dot.
    visualization_msgs::Marker marker_l;
    marker_l.header.frame_id = "cassie/left_toe";
    marker_l.header.stamp = ros::Time();
    marker_l.ns = "my_namespace";
    marker_l.id = 0;
    marker_l.type = visualization_msgs::Marker::SPHERE;
    marker_l.action = visualization_msgs::Marker::ADD;
    marker_l.pose.position.x = 0;
    marker_l.pose.position.y = 0.04;
    marker_l.pose.position.z = 0;
    marker_l.pose.orientation.x = 0.0;
    marker_l.pose.orientation.y = 0.0;
    marker_l.pose.orientation.z = 0.0;
    marker_l.pose.orientation.w = 1.0;
    marker_l.scale.x = 0.05;
    marker_l.scale.y = 0.05;
    marker_l.scale.z = 0.05;
    marker_l.color.a = 1.0; // Don't forget to set the alpha!
    marker_l.color.r = 1.0;
    marker_l.color.g = 0.0;
    marker_l.color.b = 1.0;
    foot_pub2.publish( marker_l);

    // Publish right foot dot.
    visualization_msgs::Marker marker_r;
    marker_r.header.frame_id = "cassie/right_toe";
    marker_r.header.stamp = ros::Time();
    marker_r.ns = "my_namespace";
    marker_r.id = 0;
    marker_r.type = visualization_msgs::Marker::SPHERE;
    marker_r.action = visualization_msgs::Marker::ADD;
    marker_r.pose.position.x = 0;
    marker_r.pose.position.y = 0.04;
    marker_r.pose.position.z = 0;
    marker_r.pose.orientation.x = 0.0;
    marker_r.pose.orientation.y = 0.0;
    marker_r.pose.orientation.z = 0.0;
    marker_r.pose.orientation.w = 1.0;
    marker_r.scale.x = 0.05;
    marker_r.scale.y = 0.05;
    marker_r.scale.z = 0.05;
    marker_r.color.a = 1.0; // Don't forget to set the alpha!
    marker_r.color.r = 0.0;
    marker_r.color.g = 0.0;
    marker_r.color.b = 1.0;
    foot_pub3.publish( marker_r);

    loop_rate.sleep();
  }
  infile.close();
  return 0;
}