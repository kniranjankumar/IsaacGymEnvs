#pragma once

#include <vector>
#include <map>
#include <Eigen/Dense>

namespace digit_description {
constexpr int kDigitNumJoints = 22;
constexpr int kDigitUpperBodyNumJoints = 8;
constexpr int kDigitLowerBodyNumJoints = 14;

extern const Eigen::Vector3d defaultDigitBaseXYZ;
extern const Eigen::Vector3d defaultDigitBaseRPY;

extern const std::vector<std::string> digitJointNames;
extern const std::vector<std::string> digitJointNamesPinocchio;

extern const std::map<std::string, double> collisionCylinderRadius;

extern const std::vector<std::string> footNames;

constexpr double digitFootLength = 0.2;
constexpr double digitFootWidth = 0.1;

constexpr double digitCloseLoopAngleCorrection = 0.;

const Eigen::VectorXd get_digit_default_joint_angles();
const Eigen::VectorXd get_digit_default_joint_angles_pin();
const Eigen::VectorXd get_digit_default_q0();
const Eigen::VectorXd get_digit_default_q0_pin();
const std::map<std::string, double> get_default_joint_angle_map();
Eigen::VectorXd get_pinocchio_joint_angles_from_drake(Eigen::VectorXd drake_joints);
Eigen::VectorXd get_pinocchio_q_from_drake(Eigen::VectorXd drake_q);

const Eigen::VectorXd get_digit_upper_body_default_joint_angles();
const Eigen::VectorXd get_digit_lower_body_default_joint_angles();
const Eigen::VectorXd get_digit_upper_body_default_q0();
const Eigen::VectorXd get_digit_lower_body_default_q0();

}