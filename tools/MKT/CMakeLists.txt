PROJECT(MKT)
#Debug
#SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE RELEASE)

#c++11
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE(MKT main.cpp BinFile.cpp feature.cpp vfc.cpp Track.cpp TBC.cpp tinfo.cpp )
TARGET_LINK_LIBRARIES(MKT ${OpenCV_LIBS} )
