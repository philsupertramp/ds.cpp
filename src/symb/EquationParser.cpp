#include "../../include/math/symb/EquationParser.h"
#include "../../include/math/symb/Operand.h"


std::shared_ptr<MathNode> EquationParser::createAST() {
  symbols.clear();
  operandStack.clear();
  operatorStack.clear();

  prevWasOperator = true;
  nextIsNegative  = false;
  auto eqParts    = splitEquation(processString);
  for(const auto& c : eqParts) {
    if(!parseSequence(c)) return nullptr;
  }

  while(!operatorStack.empty()) {
    auto stackTop = operatorStack[operatorStack.size() - 1];
    operatorStack.pop_back();

    auto currentOp = GetOperator(stackTop);
    if(currentOp != nullptr) {
      if(currentOp->connectionType == NodeConnectionType::ConnectionType_Dual) {
        currentOp->right = operandStack[operandStack.size() - 1];
        operandStack.pop_back();
      }

      currentOp->left = operandStack[operandStack.size() - 1];
      operandStack.pop_back();

      operandStack.push_back(currentOp);
    }
  }

  return !operandStack.empty() ? operandStack.front() : nullptr;
}

void EquationParser::processCurrentOP(
const std::shared_ptr<Operator>& currentOp,
std::vector<std::string>& operatorStack,
std::vector<std::shared_ptr<MathNode>>& operandStack) {
  // early exit
  if(operatorStack.empty()) return;
  auto stackTop = GetOperator(operatorStack[operatorStack.size() - 1]);
  while(stackTop != nullptr && stackTop->priority >= currentOp->priority && stackTop->value[0] != '(') {
    operatorStack.pop_back();
    if(stackTop->connectionType == NodeConnectionType::ConnectionType_None) {
      if(!operatorStack.empty()) stackTop = GetOperator(operatorStack[operatorStack.size() - 1]);
      continue;
    }

    if(stackTop->connectionType == NodeConnectionType::ConnectionType_Dual) {
      // inverse order, first right then left
      stackTop->right = operandStack[operandStack.size() - 1];

      operandStack.pop_back();
    }
    stackTop->left           = operandStack[operandStack.size() - 1];
    stackTop->hasParentheses = true;

    operandStack.pop_back();
    operandStack.push_back(stackTop);
    if(!operatorStack.empty()) {
      stackTop = GetOperator(operatorStack[operatorStack.size() - 1]);
    } else {
      break;
    }
  }
}
void EquationParser::rearrangeStack(
std::vector<std::string>& operatorStack, std::vector<std::shared_ptr<MathNode>>& operandStack) {
  auto stackTop = operatorStack[operatorStack.size() - 1];
  while(!isParenthesesOpen(stackTop)) {
    // handle operator
    auto currentOp = GetOperator(stackTop);
    if(currentOp != nullptr) {
      if(currentOp->connectionType == NodeConnectionType::ConnectionType_Dual) {
        currentOp->right = operandStack[operandStack.size() - 1];
        operandStack.pop_back();
      }
      currentOp->left = operandStack[operandStack.size() - 1];
      operandStack.pop_back();

      operandStack.push_back(currentOp);
    }
    if(operatorStack.size() > 1) {
      operatorStack.pop_back();

      stackTop = operatorStack[operatorStack.size() - 1];
    } else
      break;
  }
  operatorStack.pop_back();
}
std::vector<std::string> EquationParser::splitEquation(const std::string& eqString) {
  std::vector<std::string> out;
  std::cmatch m;

  auto numberRegex   = GetRegex(MathNodeType::NodeType_Numeric);
  auto symbolRegex   = GetRegex(MathNodeType::NodeType_Symbolic);
  auto operatorRegex = GetRegex(MathNodeType::NodeType_Operator_or_Parentheses);

  auto splitEq = split(eqString, ' ');

  // bigger as -x?
  if(splitEq.size() == 1) {
    out = splitFunctionsOrElementwise(splitEq[0]);
  } else {
    for(const auto& elem : splitEq) {
      // only allow full matches
      if((elem.size() == 1)
               || (
               (std::regex_match(elem.c_str(), m, symbolRegex) && m.str().size() == elem.size())
               || (std::regex_match(elem.c_str(), m, numberRegex) && m.str().size() == elem.size())
               || (std::regex_match(elem.c_str(), m, operatorRegex) && m.str().size() == elem.size())
               )){
        out.push_back(elem);
      } else {
        auto withFun = splitFunctionsOrElementwise(elem);
        out.insert(out.end(), withFun.begin(), withFun.end());
      }
    }
  }

  return out;
}
std::vector<std::string> EquationParser::extractObjects(std::string& eq, const std::vector<std::string>& container) {
  std::vector<std::string> out;
  bool keepRunning = true;
  while(keepRunning) {
    for(const auto& funStr : container) {
      auto start  = eq.find(funStr);
      keepRunning = false;
      if(start != std::string::npos) {
        if(start > 0) {
          auto ls = splitFunctionsOrElementwise(eq.substr(0, start));
          out.insert(out.end(), ls.begin(), ls.end());
        }
        out.push_back(funStr);
        auto endPos = start + funStr.size();
        if(endPos < eq.size()) {
          // remaining right side, continue
          eq          = eq.substr(endPos, eq.size() - endPos);
          keepRunning = true;
        } else {
          // processing done, return
          keepRunning = false;
        }
      }
    }
  }
  return out;
}
