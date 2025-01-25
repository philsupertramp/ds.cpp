/**
 * @file
 *
 * Missing string manipulations.
 *
 * This header file contains essential methods to manipulate strings.
 */
#pragma once
#include <cstdarg>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

/**
 * missing string printf
 * this is safe and convenient but not exactly efficient
 * @param fmt string to be formatted
 * @param ... variable amount of parameters
 * @returns formatted string
 */
inline std::string format(const char* fmt, ...) {
  int size     = 512;
  char* buffer = 0;
  buffer       = new char[size];
  va_list vl;
  va_start(vl, fmt);
  int nsize = vsnprintf(buffer, size, fmt, vl);
  std::string ret(buffer);
  if(size <= nsize) { //fail delete buffer and try again
    delete[] buffer;
    buffer = 0;
    buffer = new char[nsize - size + 2]; //+1 for /0
    nsize  = vsnprintf(buffer, (nsize - size) + 2, &fmt[size], vl);
    ret += buffer;
  }
  va_end(vl);
  delete[] buffer;
  return ret;
}

/**
 * Strips whitespaces of a string
 * @param in string which may contain whitespaces
 * @returns input string without whitespaces
 */
std::string strip(const std::string& in);

/**
 * from https://stackoverflow.com/a/236803
 * ---------------------------------------------- **/

/**
 * splits s into result by delim
 * @tparam Out
 * @param s
 * @param delim
 * @param result
 */
template<typename Out>
void split(const std::string& s, char delim, Out result);


/**
 * Helper method to simplify call to split
 * @param s string to split
 * @param delim character to split by
 * @returns vector of strings split' by delim
 */
std::vector<std::string> split(const std::string& s, char delim);

/**
 * Split after each character
 * @tparam Out container type to write to
 * @param s string to split
 * @param result container to write results in
 */
template<typename Out>
void split(const std::string& s, Out result);


/**
 * Helper method to simplify call to split
 * @param s string to split
 * @returns split vector of strings
 */
std::vector<std::string> split(const std::string& s);

std::vector<std::string> split_by_regex(const std::string& s, const std::regex& regex);

/** ----------------------------------------------- **/

/**
 * \example TestFormat.cpp
 * This is an example on how to use the format.h file.
 */