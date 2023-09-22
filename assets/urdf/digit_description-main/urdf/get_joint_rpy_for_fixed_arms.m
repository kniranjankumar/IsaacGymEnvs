% To modify urdf for fixed-arms configuration
% Author: William Pao. 10/13/2022

clear; clc;

%% 1st method: requires Robotics System toolbox

% rpy_old = [-1.57079632679 -1.3962633 1.57079632679];
% ypr_old = fliplr(rpy_old);
% R_old = eul2rotm(ypr_old);
% 
% q_desired = -0.150623;
% R_q_desired = eul2rotm([q_desired, 0, 0]); % rotation around +z axis
% 
% R_new = R_old * R_q_desired;
% ypr_new = rotm2eul(R_new);
% rpy_new = fliplr(ypr_new);
% 
% display(rpy_new);

%% 2nd method: requires William's "Rot" and "R2rpy" function

rpy_old = [-1.57079632679 -0.3926991 0];
ypr_old = fliplr(rpy_old);
R_old = Rot(ypr_old, 'ZYX');

q_desired = 0.141287;
R_q_desired = Rot(q_desired, 'Z'); % rotation around +z axis % !!! axis for shoulder pitch joint is -z axis

R_new = R_old * R_q_desired;
rpy_new = R2rpy(R_new);

clc;
display(rpy_new);

function R = Rot (angles, directions)
% Get rotational matrix
% (Modified from rotx/y/z since it is confusing to use angles in degrees)
% angles: array of angles in degrees.
% directions: a string combination of x y z
% Author: William Pao. 05/24/2022

assert(length(angles)==length(directions),'Error in function Rot: Length of angles and directions must match.');

R = eye(3); % initialize R matrix with a 3x3 identity matrix

for i = 1:length(angles)
    angle = angles(i);
    switch lower(directions(i))
        case 'x'
            R = R * [1 0 0; 0 cos(angle) -sin(angle); 0 sin(angle) cos(angle)];
        case 'y'
            R = R * [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)];
        case 'z'
            R = R * [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1];
        otherwise
            error('Error in function Rot: rotation direction can only be x y or z.');
    end
    
end

end


function rpy = R2rpy (R)
% Get roll pitch yaw from Rotational Matrix
% Author: William Pao. 10/13/2022

assert(all(size(R)==3), 'Error in function R2rpy: Rotation Matrix must be a 3x3 matrix.');

roll = atan2(R(3,2), R(3,3));
pitch = -asin(R(3,1));
yaw = atan2(R(2,1), R(1,1));

rpy = [roll, pitch, yaw];

end