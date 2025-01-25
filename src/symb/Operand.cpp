#include "../../include/math/symb/Operand.h"

bool isConstant(const std::string& in) {
  std::unordered_set<std::string> funList;
  for(const auto& elem : DefaultSymbols) { funList.insert(elem.first); }
  return funList.find(in) != funList.end();
}
bool hasSymbol(const std::vector<std::shared_ptr<Symbolic>>& container, const std::shared_ptr<Symbolic>& sym) {
  for(const auto& elem : container) {
    if(bool(strcmp(elem->value, sym->value) == 0) && elem->isNegative == sym->isNegative) return true;
  }
  return false;
}
