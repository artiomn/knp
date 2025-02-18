# ------------------------- Begin Generic CMake Variable Logging ------------------

# /*    C++ comment style not allowed.    */


# If you are building in-source, this is the same as CMAKE_SOURCE_DIR, otherwise
# this is the top level directory of your build tree.
message( STATUS "CMAKE_BINARY_DIR:         " ${CMAKE_BINARY_DIR} )

# If you are building in-source, this is the same as CMAKE_CURRENT_SOURCE_DIR, otherwise this
# is the directory where the compiled or generated files from the current CMakeLists.txt will go to.
message( STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

# This is the directory, from which CMake is started, i.e. the top level source directory.
message( STATUS "CMAKE_SOURCE_DIR:         " ${CMAKE_SOURCE_DIR} )

# This is the directory where the currently processed CMakeLists.txt is located in.
message( STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )

# Contains the full path to the top level directory of your build tree.
message( STATUS "PROJECT_BINARY_DIR: " ${PROJECT_BINARY_DIR} )

# Contains the full path to the root of your project source directory,
# i.e. to the nearest directory where CMakeLists.txt contains the project() command.
message( STATUS "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )

# Set this variable to specify a common place where CMake should put all executable files
# (instead of CMAKE_CURRENT_BINARY_DIR).
message( STATUS "EXECUTABLE_OUTPUT_PATH: " ${EXECUTABLE_OUTPUT_PATH} )

# Set this variable to specify a common place where CMake should put all libraries
# (instead of CMAKE_CURRENT_BINARY_DIR).
message( STATUS "LIBRARY_OUTPUT_PATH:     " ${LIBRARY_OUTPUT_PATH} )

# Tell CMake to search first in directories listed in CMAKE_MODULE_PATH
# when you use find_package() or include().
message( STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH} )

# This is the complete path of the CMake which runs currently (e.g. /usr/local/bin/cmake).
message( STATUS "CMAKE_COMMAND: " ${CMAKE_COMMAND} )

# This is the CMake installation directory.
message( STATUS "CMAKE_ROOT: " ${CMAKE_ROOT} )

# This is the filename including the complete path of the file where this variable is used.
message( STATUS "CMAKE_CURRENT_LIST_FILE: " ${CMAKE_CURRENT_LIST_FILE} )

# This is linenumber where the variable is used.
message( STATUS "CMAKE_CURRENT_LIST_LINE: " ${CMAKE_CURRENT_LIST_LINE} )

# This is used when searching for include files e.g. using the find_path() command.
message( STATUS "CMAKE_INCLUDE_PATH: " ${CMAKE_INCLUDE_PATH} )

# This is used when searching for libraries e.g. using the find_library() command.
message( STATUS "CMAKE_LIBRARY_PATH: " ${CMAKE_LIBRARY_PATH} )

# The complete system name, e.g. "Linux-2.4.22", "FreeBSD-5.4-RELEASE" or "Windows 5.1".
message( STATUS "CMAKE_SYSTEM: " ${CMAKE_SYSTEM} )

# The short system name, e.g. "Linux", "FreeBSD" or "Windows".
message( STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME} )

# Only the version part of CMAKE_SYSTEM.
message( STATUS "CMAKE_SYSTEM_VERSION: " ${CMAKE_SYSTEM_VERSION} )

# The processor name (e.g. "Intel(R) Pentium(R) M processor 2.00GHz").
message( STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR} )

# This is TRUE on all UNIX-like OS's, including Apple OS X and CygWin.
message( STATUS "UNIX: " ${UNIX} )

# This is TRUE on Windows, including CygWin.
message( STATUS "WIN32: " ${WIN32} )

# This is TRUE on Apple OS X.
message( STATUS "APPLE: " ${APPLE} )

# This is TRUE when using the MinGW compiler in Windows.
message( STATUS "MINGW: " ${MINGW} )

# This is TRUE on Windows when using the CygWin version of cmake.
message( STATUS "CYGWIN: " ${CYGWIN} )

# This is TRUE on Windows when using a Borland compiler.
message( STATUS "BORLAND: " ${BORLAND} )

# Microsoft compiler.
message( STATUS "MSVC: " ${MSVC} )
message( STATUS "MSVC_IDE: " ${MSVC_IDE} )
message( STATUS "MSVC60: " ${MSVC60} )
message( STATUS "MSVC70: " ${MSVC70} )
message( STATUS "MSVC71: " ${MSVC71} )
message( STATUS "MSVC80: " ${MSVC80} )
message( STATUS "CMAKE_COMPILER_2005: " ${CMAKE_COMPILER_2005} )


# Set this to TRUE if you don't want to rebuild the object files if the rules have changed,
# but not the actual source files or headers (e.g. if you changed the some compiler switches).
message( STATUS "CMAKE_SKIP_RULE_DEPENDENCY: " ${CMAKE_SKIP_RULE_DEPENDENCY} )

# Since CMake 2.1 the install rule depends on all, i.e. everything will be built before installing.
# If you don't like this, set this one to TRUE.
message( STATUS "CMAKE_SKIP_INSTALL_ALL_DEPENDENCY: " ${CMAKE_SKIP_INSTALL_ALL_DEPENDENCY} )

# If set, runtime paths are not added when using shared libraries. Default it is set to OFF.
message( STATUS "CMAKE_SKIP_RPATH: " ${CMAKE_SKIP_RPATH} )

# Set this to TRUE if you are using makefiles and want to see the full compile and link
# commands instead of only the shortened ones.
message( STATUS "CMAKE_VERBOSE_MAKEFILE: " ${CMAKE_VERBOSE_MAKEFILE} )

# This will cause CMake to not put in the rules that re-run CMake. This might be useful if
# you want to use the generated build files on another machine.
message( STATUS "CMAKE_SUPPRESS_REGENERATION: " ${CMAKE_SUPPRESS_REGENERATION} )


# A simple way to get switches to the compiler is to use add_definitions().
# But there are also two variables exactly for this purpose:

# The compiler flags for compiling C sources.
message( STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS} )

# The compiler flags for compiling C++ sources.
message( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )


# Choose the type of build.  Example: set(CMAKE_BUILD_TYPE Debug).
message( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

# If this is set to ON, then all libraries are built as shared libraries by default.
message( STATUS "BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS} )

# The compiler used for C files.
message( STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )

# The compiler used for C++ files.
message( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )

# If the compiler is a variant of gcc, this should be set to 1.
message( STATUS "CMAKE_COMPILER_IS_GNUCC: " ${CMAKE_COMPILER_IS_GNUCC} )

# If the compiler is a variant of g++, this should be set to 1.
message( STATUS "CMAKE_COMPILER_IS_GNUCXX : " ${CMAKE_COMPILER_IS_GNUCXX} )

# The tools for creating libraries.
message( STATUS "CMAKE_AR: " ${CMAKE_AR} )
message( STATUS "CMAKE_RANLIB: " ${CMAKE_RANLIB} )

#
#message( STATUS ": " ${} )

# ------------------------- End of Generic CMake Variable Logging ------------------
