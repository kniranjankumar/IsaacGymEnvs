#include "digit_description/digit_constants.h"

namespace digit_description {
const Eigen::Vector3d defaultDigitBaseXYZ{0, 0, 1.015};
const Eigen::Vector3d defaultDigitBaseRPY{0, 0, 0};

const std::vector<std::string> digitJointNames{
    "hip_abduction_left", "hip_rotation_left", "hip_flexion_left", "knee_joint_left", 
    "shin_to_tarsus_left", "toe_pitch_joint_left", "toe_roll_joint_left", "shoulder_roll_joint_left", 
    "shoulder_pitch_joint_left", "shoulder_yaw_joint_left", "elbow_joint_left",
    "hip_abduction_right", "hip_rotation_right", "hip_flexion_right", "knee_joint_right", 
    "shin_to_tarsus_right", "toe_pitch_joint_right", "toe_roll_joint_right", "shoulder_roll_joint_right", 
    "shoulder_pitch_joint_right", "shoulder_yaw_joint_right", "elbow_joint_right"
};

const std::vector<std::string> digitJointNamesPinocchio{
    "hip_abduction_left", "hip_rotation_left", "hip_flexion_left", "knee_joint_left", 
    "shin_to_tarsus_left", "toe_pitch_joint_left", "toe_roll_joint_left", 
    "hip_abduction_right", "hip_rotation_right", "hip_flexion_right", "knee_joint_right", 
    "shin_to_tarsus_right", "toe_pitch_joint_right", "toe_roll_joint_right", 
    "shoulder_roll_joint_left", "shoulder_pitch_joint_left", "shoulder_yaw_joint_left", "elbow_joint_left",
    "shoulder_roll_joint_right", 
    "shoulder_pitch_joint_right", "shoulder_yaw_joint_right", "elbow_joint_right"
};

const std::vector<std::string> digitUpperBodyJointNamesPinocchio{
    "hip_abduction_left", "hip_rotation_left", "hip_flexion_left", "knee_joint_left", 
    "shin_to_tarsus_left", "toe_pitch_joint_left", "toe_roll_joint_left", 
    "hip_abduction_right", "hip_rotation_right", "hip_flexion_right", "knee_joint_right", 
    "shin_to_tarsus_right", "toe_pitch_joint_right", "toe_roll_joint_right", 
    "shoulder_roll_joint_left", "shoulder_pitch_joint_left", "shoulder_yaw_joint_left", "elbow_joint_left",
    "shoulder_roll_joint_right", 
    "shoulder_pitch_joint_right", "shoulder_yaw_joint_right", "elbow_joint_right"
};

const std::map<std::string, double> collisionCylinderRadius{
    {"torso", 0.15},
    {"left_hip_roll", 0.04}, {"left_hip_pitch", 0.07}, {"left_knee", 0.04},
    {"left_shin", 0.06}, {"left_tarsus", 0.05}, {"left_toe_roll", 0.05}, 
    {"left_shoulder_pitch", 0.05}, {"left_elbow", 0.05}, 
    {"right_hip_roll", 0.04}, {"right_hip_pitch", 0.07}, {"right_knee", 0.04},
    {"right_shin", 0.06}, {"right_tarsus", 0.05}, {"right_toe_roll", 0.05}, 
    {"right_shoulder_pitch", 0.05}, {"right_elbow", 0.05}
};

const std::vector<std::string> footNames{"left_foot_bottom", "right_foot_bottom"};

const Eigen::VectorXd get_digit_default_joint_angles() {
    Eigen::VectorXd default_joint_angles(kDigitNumJoints);
    default_joint_angles << 
        0.3605, -0.0076, 0.3399, 0.3, -0.3, 0.23244, 0, -0.132, 0.895, -0.005, -0.5,
        -0.3605, 0.0076, -0.3399, -0.3, 0.3, -0.23244, 0, 0.132, -0.895, 0.005, 0.5;
    return default_joint_angles;
}

const Eigen::VectorXd get_digit_default_joint_angles_pin() {
    Eigen::VectorXd joint_angles_pin(kDigitNumJoints);
    auto map = get_default_joint_angle_map();

    for (size_t i = 0; i < joint_angles_pin.size(); i++) {
        joint_angles_pin[i] = map[digitJointNamesPinocchio[i]];
    }

    return joint_angles_pin;
}

const Eigen::VectorXd get_digit_default_q0() {
    Eigen::VectorXd q0(kDigitNumJoints+7);   
    q0.head(4) = Eigen::Vector4d{1, 0, 0, 0};
    q0.segment(4, 3) = defaultDigitBaseXYZ;
    q0.tail(kDigitNumJoints) = get_digit_default_joint_angles();
    return q0;
}

const Eigen::VectorXd get_digit_default_q0_pin() {
    Eigen::VectorXd q0(kDigitNumJoints+7);   
    q0.segment(3, 4) = Eigen::Vector4d{0, 0, 0, 1};
    q0.head(3) = defaultDigitBaseXYZ;
    q0.tail(kDigitNumJoints) = get_digit_default_joint_angles_pin();
    return q0;
}

const std::map<std::string, double> get_default_joint_angle_map() {
    std::map<std::string, double> map;
    auto joint_angles = get_digit_default_joint_angles();
    for (size_t i = 0; i < joint_angles.size(); i++) {
        map[digitJointNames[i]] = joint_angles[i];
    }

    return map;
}

Eigen::VectorXd get_pinocchio_joint_angles_from_drake(Eigen::VectorXd drake_joints) {    
    std::map<std::string, double> map;
    for (size_t i = 0; i < drake_joints.size(); i++) {
        map[digitJointNames[i]] = drake_joints[i];
    }

    Eigen::VectorXd joint_angles_pin(kDigitNumJoints);

    for (size_t i = 0; i < joint_angles_pin.size(); i++) {
        joint_angles_pin[i] = map[digitJointNamesPinocchio[i]];
    }

    return joint_angles_pin;
}

Eigen::VectorXd get_pinocchio_q_from_drake(Eigen::VectorXd drake_q) {
    Eigen::VectorXd q{kDigitNumJoints+7};
    q.head(3) = drake_q.segment(4, 3); // x y z
    q.segment(3, 3) = drake_q.segment(1, 3); // qx qy qz
    q[6] = drake_q[0];
    q.tail(kDigitNumJoints) = get_pinocchio_joint_angles_from_drake(drake_q.tail(kDigitNumJoints));
    return q;
}

const Eigen::VectorXd get_digit_upper_body_default_joint_angles() {
    return get_digit_default_joint_angles_pin().tail(kDigitUpperBodyNumJoints);
}

const Eigen::VectorXd get_digit_lower_body_default_joint_angles() {
    return get_digit_default_joint_angles_pin().head(kDigitLowerBodyNumJoints);
}

const Eigen::VectorXd get_digit_upper_body_default_q0() {
    Eigen::VectorXd q0{kDigitUpperBodyNumJoints + 7};
    q0.segment(3, 4) = Eigen::Vector4d{0, 0, 0, 1};
    q0.head(3) = defaultDigitBaseXYZ;
    q0.tail(kDigitUpperBodyNumJoints) = get_digit_upper_body_default_joint_angles();
    return q0;
}

const Eigen::VectorXd get_digit_lower_body_default_q0() {
    Eigen::VectorXd q0{kDigitLowerBodyNumJoints + 7};
    q0.segment(3, 4) = Eigen::Vector4d{0, 0, 0, 1};
    q0.head(3) = defaultDigitBaseXYZ;
    q0.tail(kDigitLowerBodyNumJoints) = get_digit_lower_body_default_joint_angles();
    return q0;
}
}