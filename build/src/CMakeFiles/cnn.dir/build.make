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
CMAKE_SOURCE_DIR = /home/lollerfirst/dev/CNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lollerfirst/dev/CNN/build

# Include any dependencies generated for this target.
include src/CMakeFiles/cnn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/cnn.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cnn.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cnn.dir/flags.make

src/CMakeFiles/cnn.dir/activation.cpp.o: src/CMakeFiles/cnn.dir/flags.make
src/CMakeFiles/cnn.dir/activation.cpp.o: ../src/activation.cpp
src/CMakeFiles/cnn.dir/activation.cpp.o: src/CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lollerfirst/dev/CNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/cnn.dir/activation.cpp.o"
	cd /home/lollerfirst/dev/CNN/build/src && /usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/cnn.dir/activation.cpp.o -MF CMakeFiles/cnn.dir/activation.cpp.o.d -o CMakeFiles/cnn.dir/activation.cpp.o -c /home/lollerfirst/dev/CNN/src/activation.cpp

src/CMakeFiles/cnn.dir/activation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/activation.cpp.i"
	cd /home/lollerfirst/dev/CNN/build/src && /usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lollerfirst/dev/CNN/src/activation.cpp > CMakeFiles/cnn.dir/activation.cpp.i

src/CMakeFiles/cnn.dir/activation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/activation.cpp.s"
	cd /home/lollerfirst/dev/CNN/build/src && /usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lollerfirst/dev/CNN/src/activation.cpp -o CMakeFiles/cnn.dir/activation.cpp.s

# Object files for target cnn
cnn_OBJECTS = \
"CMakeFiles/cnn.dir/activation.cpp.o"

# External object files for target cnn
cnn_EXTERNAL_OBJECTS =

src/libcnn.a: src/CMakeFiles/cnn.dir/activation.cpp.o
src/libcnn.a: src/CMakeFiles/cnn.dir/build.make
src/libcnn.a: src/CMakeFiles/cnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lollerfirst/dev/CNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcnn.a"
	cd /home/lollerfirst/dev/CNN/build/src && $(CMAKE_COMMAND) -P CMakeFiles/cnn.dir/cmake_clean_target.cmake
	cd /home/lollerfirst/dev/CNN/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cnn.dir/build: src/libcnn.a
.PHONY : src/CMakeFiles/cnn.dir/build

src/CMakeFiles/cnn.dir/clean:
	cd /home/lollerfirst/dev/CNN/build/src && $(CMAKE_COMMAND) -P CMakeFiles/cnn.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cnn.dir/clean

src/CMakeFiles/cnn.dir/depend:
	cd /home/lollerfirst/dev/CNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lollerfirst/dev/CNN /home/lollerfirst/dev/CNN/src /home/lollerfirst/dev/CNN/build /home/lollerfirst/dev/CNN/build/src /home/lollerfirst/dev/CNN/build/src/CMakeFiles/cnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cnn.dir/depend

