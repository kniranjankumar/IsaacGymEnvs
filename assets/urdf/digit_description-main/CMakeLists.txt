cmake_minimum_required(VERSION 3.5.1)
project(digit_description)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf
)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

catkin_package(
  INCLUDE_DIRS 
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  LIBRARIES 
  ${PROJECT_NAME} 
  CATKIN_DEPENDS 
  ${CATKIN_PACKAGE_DEPENDENCIES}
)

include_directories(${catkin_INCLUDE_DIRS}
  include
  ${EIGEN3_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME}
    src/digit_constants.cpp
)
target_link_libraries(
    ${PROJECT_NAME}
    ${catkin_LIBRARIES} 
)

add_executable(viz_lbl src/viz_lbl.cpp)
target_link_libraries(viz_lbl ${catkin_LIBRARIES} Eigen3::Eigen)


# if(CATKIN_ENABLE_TESTING)
#   find_package(roslaunch REQUIRED)
#   roslaunch_add_file_check(launch)
# endif()

install(DIRECTORY digit-v3 launch rviz urdf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)