# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dartisan/Documents/dev/Python-RVO2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dartisan/Documents/dev/Python-RVO2

# Include any dependencies generated for this target.
include src/CMakeFiles/RVO.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/RVO.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/RVO.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/RVO.dir/flags.make

src/CMakeFiles/RVO.dir/Agent.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/Agent.cpp.o: src/Agent.cpp
src/CMakeFiles/RVO.dir/Agent.cpp.o: src/CMakeFiles/RVO.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dartisan/Documents/dev/Python-RVO2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/RVO.dir/Agent.cpp.o"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/RVO.dir/Agent.cpp.o -MF CMakeFiles/RVO.dir/Agent.cpp.o.d -o CMakeFiles/RVO.dir/Agent.cpp.o -c /home/dartisan/Documents/dev/Python-RVO2/src/Agent.cpp

src/CMakeFiles/RVO.dir/Agent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/Agent.cpp.i"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dartisan/Documents/dev/Python-RVO2/src/Agent.cpp > CMakeFiles/RVO.dir/Agent.cpp.i

src/CMakeFiles/RVO.dir/Agent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/Agent.cpp.s"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dartisan/Documents/dev/Python-RVO2/src/Agent.cpp -o CMakeFiles/RVO.dir/Agent.cpp.s

src/CMakeFiles/RVO.dir/KdTree.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/KdTree.cpp.o: src/KdTree.cpp
src/CMakeFiles/RVO.dir/KdTree.cpp.o: src/CMakeFiles/RVO.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dartisan/Documents/dev/Python-RVO2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/RVO.dir/KdTree.cpp.o"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/RVO.dir/KdTree.cpp.o -MF CMakeFiles/RVO.dir/KdTree.cpp.o.d -o CMakeFiles/RVO.dir/KdTree.cpp.o -c /home/dartisan/Documents/dev/Python-RVO2/src/KdTree.cpp

src/CMakeFiles/RVO.dir/KdTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/KdTree.cpp.i"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dartisan/Documents/dev/Python-RVO2/src/KdTree.cpp > CMakeFiles/RVO.dir/KdTree.cpp.i

src/CMakeFiles/RVO.dir/KdTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/KdTree.cpp.s"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dartisan/Documents/dev/Python-RVO2/src/KdTree.cpp -o CMakeFiles/RVO.dir/KdTree.cpp.s

src/CMakeFiles/RVO.dir/Obstacle.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/Obstacle.cpp.o: src/Obstacle.cpp
src/CMakeFiles/RVO.dir/Obstacle.cpp.o: src/CMakeFiles/RVO.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dartisan/Documents/dev/Python-RVO2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/RVO.dir/Obstacle.cpp.o"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/RVO.dir/Obstacle.cpp.o -MF CMakeFiles/RVO.dir/Obstacle.cpp.o.d -o CMakeFiles/RVO.dir/Obstacle.cpp.o -c /home/dartisan/Documents/dev/Python-RVO2/src/Obstacle.cpp

src/CMakeFiles/RVO.dir/Obstacle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/Obstacle.cpp.i"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dartisan/Documents/dev/Python-RVO2/src/Obstacle.cpp > CMakeFiles/RVO.dir/Obstacle.cpp.i

src/CMakeFiles/RVO.dir/Obstacle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/Obstacle.cpp.s"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dartisan/Documents/dev/Python-RVO2/src/Obstacle.cpp -o CMakeFiles/RVO.dir/Obstacle.cpp.s

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o: src/CMakeFiles/RVO.dir/flags.make
src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o: src/RVOSimulator.cpp
src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o: src/CMakeFiles/RVO.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dartisan/Documents/dev/Python-RVO2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o -MF CMakeFiles/RVO.dir/RVOSimulator.cpp.o.d -o CMakeFiles/RVO.dir/RVOSimulator.cpp.o -c /home/dartisan/Documents/dev/Python-RVO2/src/RVOSimulator.cpp

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RVO.dir/RVOSimulator.cpp.i"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dartisan/Documents/dev/Python-RVO2/src/RVOSimulator.cpp > CMakeFiles/RVO.dir/RVOSimulator.cpp.i

src/CMakeFiles/RVO.dir/RVOSimulator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RVO.dir/RVOSimulator.cpp.s"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dartisan/Documents/dev/Python-RVO2/src/RVOSimulator.cpp -o CMakeFiles/RVO.dir/RVOSimulator.cpp.s

# Object files for target RVO
RVO_OBJECTS = \
"CMakeFiles/RVO.dir/Agent.cpp.o" \
"CMakeFiles/RVO.dir/KdTree.cpp.o" \
"CMakeFiles/RVO.dir/Obstacle.cpp.o" \
"CMakeFiles/RVO.dir/RVOSimulator.cpp.o"

# External object files for target RVO
RVO_EXTERNAL_OBJECTS =

src/libRVO.a: src/CMakeFiles/RVO.dir/Agent.cpp.o
src/libRVO.a: src/CMakeFiles/RVO.dir/KdTree.cpp.o
src/libRVO.a: src/CMakeFiles/RVO.dir/Obstacle.cpp.o
src/libRVO.a: src/CMakeFiles/RVO.dir/RVOSimulator.cpp.o
src/libRVO.a: src/CMakeFiles/RVO.dir/build.make
src/libRVO.a: src/CMakeFiles/RVO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dartisan/Documents/dev/Python-RVO2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libRVO.a"
	cd /home/dartisan/Documents/dev/Python-RVO2/src && $(CMAKE_COMMAND) -P CMakeFiles/RVO.dir/cmake_clean_target.cmake
	cd /home/dartisan/Documents/dev/Python-RVO2/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RVO.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/RVO.dir/build: src/libRVO.a
.PHONY : src/CMakeFiles/RVO.dir/build

src/CMakeFiles/RVO.dir/clean:
	cd /home/dartisan/Documents/dev/Python-RVO2/src && $(CMAKE_COMMAND) -P CMakeFiles/RVO.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/RVO.dir/clean

src/CMakeFiles/RVO.dir/depend:
	cd /home/dartisan/Documents/dev/Python-RVO2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dartisan/Documents/dev/Python-RVO2 /home/dartisan/Documents/dev/Python-RVO2/src /home/dartisan/Documents/dev/Python-RVO2 /home/dartisan/Documents/dev/Python-RVO2/src /home/dartisan/Documents/dev/Python-RVO2/src/CMakeFiles/RVO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/RVO.dir/depend

