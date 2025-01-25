#include "../include/math/format.h"
#include <iostream>


std::string strip(const std::string& in) {
  std::string out = in;
  auto space      = out.find(' ');
  while(space != std::string::npos) {
    auto backup = out.substr(0, space);
    backup += out.substr(space + 1, out.size() - backup.size());
    out   = backup;
    space = out.find(' ');
  }
  return out;
}


template<typename Out>
void split(const std::string& s, char delim, Out result) {
  std::istringstream iss(s);
  std::string item;
  while(std::getline(iss, item, delim)) { *result++ = item; }
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

template<typename Out>
void split(const std::string& s, Out result) {
  std::istringstream iss(s);
  for(auto item : s) { *result++ = std::string(1, item); }
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> elems;
  split(s, std::back_inserter(elems));
  return elems;
}
std::vector<std::string> split_by_regex(const std::string& s, const std::regex& regex) {
  std::vector<std::string> elems;
  auto copyString                 = s;
  std::sregex_token_iterator iter = std::sregex_token_iterator(copyString.begin(), copyString.end(), regex, -1);
  if((*iter).str().empty()) { ++iter; }
  std::sregex_token_iterator end;
  while(!copyString.empty() && iter != end) {
    auto sub = (*iter).str();
    if(sub.size() == 0) break;
    auto pos = copyString.find(sub);
    if(pos > 0) { elems.push_back(copyString.substr(0, pos)); }
    elems.push_back(sub);
    copyString = copyString.substr(pos + sub.size(), copyString.size() - (pos + sub.size()));
    iter       = std::sregex_token_iterator(copyString.begin(), copyString.end(), regex, -1);
    if((*iter).str().empty()) { ++iter; }
  }
  // remaining operators?
  if(!copyString.empty()) {
    auto pasta = split(copyString);
    elems.insert(elems.end(), pasta.begin(), pasta.end());
  }
  return elems;
}
