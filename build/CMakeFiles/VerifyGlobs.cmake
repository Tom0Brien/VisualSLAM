# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.20
cmake_policy(SET CMP0009 NEW)

# SRC_BIN at CMakeLists.txt:66 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "C:/msys64/home/Tom/VisualSLAM/src/*.cpp")
set(OLD_GLOB
  "C:/msys64/home/Tom/VisualSLAM/src/SLAM.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/SLAMold.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/calibrate.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/cameraModel.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/fmin.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/gaussian.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/imagefeatures.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/main.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/model.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/plot.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/settings.cpp"
  "C:/msys64/home/Tom/VisualSLAM/src/utility.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "C:/msys64/home/Tom/VisualSLAM/build/CMakeFiles/cmake.verify_globs")
endif()

# SRC_BIN at CMakeLists.txt:66 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "C:/msys64/home/Tom/VisualSLAM/src/*.h")
set(OLD_GLOB
  "C:/msys64/home/Tom/VisualSLAM/src/SLAM.h"
  "C:/msys64/home/Tom/VisualSLAM/src/calibrate.h"
  "C:/msys64/home/Tom/VisualSLAM/src/imagefeatures.h"
  "C:/msys64/home/Tom/VisualSLAM/src/plot.h"
  "C:/msys64/home/Tom/VisualSLAM/src/settings.h"
  "C:/msys64/home/Tom/VisualSLAM/src/utility.h"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "C:/msys64/home/Tom/VisualSLAM/build/CMakeFiles/cmake.verify_globs")
endif()

# SRC_BIN at CMakeLists.txt:66 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "C:/msys64/home/Tom/VisualSLAM/src/*.hpp")
set(OLD_GLOB
  "C:/msys64/home/Tom/VisualSLAM/src/cameraModel.hpp"
  "C:/msys64/home/Tom/VisualSLAM/src/fmin.hpp"
  "C:/msys64/home/Tom/VisualSLAM/src/gaussian.hpp"
  "C:/msys64/home/Tom/VisualSLAM/src/model.hpp"
  "C:/msys64/home/Tom/VisualSLAM/src/rotation.hpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "C:/msys64/home/Tom/VisualSLAM/build/CMakeFiles/cmake.verify_globs")
endif()

# SRC_TEST at CMakeLists.txt:73 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "C:/msys64/home/Tom/VisualSLAM/test/src/*.cpp")
set(OLD_GLOB
  "C:/msys64/home/Tom/VisualSLAM/test/src/main.cpp"
  "C:/msys64/home/Tom/VisualSLAM/test/src/plot.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "C:/msys64/home/Tom/VisualSLAM/build/CMakeFiles/cmake.verify_globs")
endif()

# SRC_TEST at CMakeLists.txt:73 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "C:/msys64/home/Tom/VisualSLAM/test/src/*.h")
set(OLD_GLOB
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "C:/msys64/home/Tom/VisualSLAM/build/CMakeFiles/cmake.verify_globs")
endif()

# SRC_TEST at CMakeLists.txt:73 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true "C:/msys64/home/Tom/VisualSLAM/test/src/*.hpp")
set(OLD_GLOB
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "C:/msys64/home/Tom/VisualSLAM/build/CMakeFiles/cmake.verify_globs")
endif()
