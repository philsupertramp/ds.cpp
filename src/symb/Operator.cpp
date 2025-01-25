#include "../../include/math/symb/Operator.h"

std::shared_ptr<Operator> GenerateOperator(OperatorType type) {
  switch(type) {
    case TYPE_ADDITION:
      return std::make_shared<Operator>(
      "+", [](double a, double b) { return a + b; }, OperatorPriority::OPClassLine);
    case TYPE_SUBTRACTION:
      return std::make_shared<Operator>(
      "-", [](double a, double b) { return a - b; }, OperatorPriority::OPClassLine);
    case TYPE_MULTIPLICATION:
      return std::make_shared<Operator>(
      "*", [](double a, double b) { return a * b; }, OperatorPriority::OPClassDot);
    case TYPE_DIVISION:
      return std::make_shared<Operator>(
      "/", [](double a, double b) { return a / b; }, OperatorPriority::OPClassDot);
    case TYPE_POWER:
      return std::make_shared<Operator>(
      "^", [](double a, double b) { return pow(a, b); }, OperatorPriority::OPClassPow);
    case TYPE_PARENTHESES_OPEN:
      {
        auto op = std::make_shared<Operator>(
        "(",
        []([[maybe_unused]] double a, [[maybe_unused]] double b) { return 0.0; },
        OperatorPriority::OPClassParentheses);
        op->connectionType = NodeConnectionType::ConnectionType_None;
        op->hasParentheses = true;
        return op;
      }
    case TYPE_PARENTHESES_CLOSE:
      {
        auto op = std::make_shared<Operator>(
        ")",
        []([[maybe_unused]] double a, [[maybe_unused]] double b) { return 0.0; },
        OperatorPriority::OPClassParentheses);
        op->connectionType = NodeConnectionType::ConnectionType_None;
        op->hasParentheses = true;
        return op;
      }
  }
  return std::make_shared<Operator>(
  "",
  []([[maybe_unused]] double a, [[maybe_unused]] double b) {
    std::cerr << "Operator not found.\n" << std::flush;
    return -1;
  },
  OperatorPriority::OPClassUnknown);
}

bool isFunction(const std::string& in) {
  std::unordered_set<std::string> funList;
  for(const auto& elem : DefaultFunctions) { funList.insert(elem.first); }
  return funList.find(in) != funList.end();
}
