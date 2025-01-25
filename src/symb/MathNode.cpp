#include "../../include/math/symb/MathNode.h"

std::regex GetRegex(MathNodeType type) {
  static const std::regex symbol_regex("([A-Za-z_]{1}[A-Za-z0-9_]?)");
  static const std::regex default_symbol_regex("([A-Za-z_]+)");
  static const std::regex number_regex("([0-9]+)");
  static const std::regex parentheses_regex("\\(([^\\)]*)\\)");
  static const std::regex operator_regex("([\\-+*/\\^]+)");
  static const std::regex operator_or_parentheses_regex("([\\(\\)\\-+*/\\^])");
  static const std::regex any_regex("([\\x00-\\x7F]+)");
  switch(type) {
    case NodeType_Symbolic: return symbol_regex;
    case NodeType_Numeric: return number_regex;
    case NodeType_Parentheses: return parentheses_regex;
    case NodeType_DefaultSymbol: return default_symbol_regex;
    case NodeType_Any: return any_regex;
    case NodeType_Operator_or_Parentheses: return operator_or_parentheses_regex;
    default: return operator_regex;
  }
}
