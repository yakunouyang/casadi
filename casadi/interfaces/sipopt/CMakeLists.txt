cmake_minimum_required(VERSION 3.10.2)

set(NLPSOL_SIPOPT_SRCS
  sipopt_interface.hpp
  sipopt_interface.cpp
  sipopt_nlp.hpp
  sipopt_nlp.cpp
  sipopt_interface_meta.cpp)

if(WITH_IPOPT_CALLBACK)
  add_definitions(-DWITH_IPOPT_CALLBACK)
endif()

casadi_plugin(Nlpsol sipopt ${NLPSOL_SIPOPT_SRCS})
casadi_plugin_link_libraries(Nlpsol sipopt ipopt sipopt)

#set_property(TARGET casadi_nlpsol_ipopt PROPERTY PKG_CONFIG_REQUIRES ipopt)
