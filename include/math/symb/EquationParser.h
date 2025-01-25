#pragma once

#include <utility>

#include "MathNode.h"
#include "Operand.h"
#include "Operator.h"


/**
 * Parses a given string representation of an equation
 * generates an AST using EquationParser::createAST
 */
class EquationParser
{
  //! equation string to parse
  std::string processString;


  //! flag for parsing process
  bool prevWasOperator = true;
  //! flag for parsing process to detect whether a node is negated
  bool nextIsNegative = false;
  //! operator stack object, used during parsing
  std::vector<std::string> operatorStack;
  //! operand stack object, used during parsing
  std::vector<std::shared_ptr<MathNode>> operandStack;

public:
  //! storage for symbols, gets cleared on EquationParser::createAST
  std::vector<std::shared_ptr<Symbolic>> symbols;

  /**
   * Default constructor
   */
  explicit EquationParser(std::string in)
    : processString(std::move(in)) { }

  /**
   * helper method to generate superset of given vectors
   * @param a set of symbols
   * @param b other set of symbols
   * @returns merged unique set of symbols
   */
  static std::vector<std::shared_ptr<Symbolic>> buildSymbolSuperSet(
  const std::vector<std::shared_ptr<Symbolic>>& a, const std::vector<std::shared_ptr<Symbolic>>& b) {
    std::vector<std::shared_ptr<Symbolic>> out = a;
    auto testIsInContainer                     = [&out](const std::shared_ptr<Symbolic>& sym) {
      for(const auto& elem : out) {
        if(bool(strcmp(elem->value, sym->value) == 0) && elem->isNegative == sym->isNegative) return true;
      }
      return false;
    };

    size_t numElements = out.size();
    for(const auto& i : b) {
      if(!testIsInContainer(i)) {
        out.push_back(i);
        numElements++;
      }
    }

    return out;
  }

  /**
   * Creates A(bstract)S(yntax)T(ree) using shunting-yard algorithm, thanks Mr. Dijkstra :)
   * https://en.wikipedia.org/wiki/Shunting-yard_algorithm
   *
   * @param processString
   * @returns
   */
  std::shared_ptr<MathNode> createAST();

  /**
   * Lexer function to determine first connecting operator
   * @param valString
   * @returns
   */
  static std::shared_ptr<Operator> GetOperator(const std::string& valString) {
    if(valString.empty()) { return {}; }
    if(valString.size() > 1) { return GetFunction(valString); }
    switch(int(strip(valString).c_str()[0])) {
      case int(OperatorValue::VALUE_MULTIPLICATION): return GenerateOperator(OperatorType::TYPE_MULTIPLICATION);
      case int(OperatorValue::VALUE_DIVISION): return GenerateOperator(OperatorType::TYPE_DIVISION);
      case int(OperatorValue::VALUE_ADDITION): return GenerateOperator(OperatorType::TYPE_ADDITION);
      case int(OperatorValue::VALUE_SUBTRACTION): return GenerateOperator(OperatorType::TYPE_SUBTRACTION);
      case int(OperatorValue::VALUE_POWER): return GenerateOperator(OperatorType::TYPE_POWER);
      case int(OperatorValue::VALUE_PARENTHESES_OPEN): return GenerateOperator(OperatorType::TYPE_PARENTHESES_OPEN);
      case int(OperatorValue::VALUE_PARENTHESES_CLOSE): return GenerateOperator(OperatorType::TYPE_PARENTHESES_CLOSE);
    }
    return {};
  }

private:
  /**
   * parses a sequence, can be a Function, Operator, Number, Symbol or ANY
   * @param c sequence to parse
   * @returns success
   */
  bool parseSequence(const std::string& c) {
    std::string copy = c;
    if(isNumber(copy)) {
      auto sym        = std::make_shared<Number>(copy);
      sym->isNegative = nextIsNegative;
      nextIsNegative  = false;
      operandStack.push_back(sym);
    } else if(isConstant(copy)) {
      auto sym        = DefaultSymbols[copy];
      sym->isNegative = nextIsNegative;
      nextIsNegative  = false;
      operandStack.push_back(sym);
    } else if(isSymbol(copy)) {
      auto symbol        = std::make_shared<Symbolic>(copy);
      symbol->isNegative = nextIsNegative;
      nextIsNegative     = false;
      operandStack.push_back(symbol);
      if(!hasSymbol(symbols, symbol)) symbols.push_back(symbol);
    } else if(isParenthesesOpen(copy) || isFunction(copy)) {
      operatorStack.push_back(copy);
    } else if(isOperator(copy)) {
      auto currentOp = GetOperator(copy);
      if(prevWasOperator && currentOp->value[0] == '-') {
        nextIsNegative = !nextIsNegative;
      } else {
        if(currentOp->value[0] == '-') {
          currentOp      = GetOperator("+");
          copy           = "+";
          nextIsNegative = !nextIsNegative;
        }
        processCurrentOP(currentOp, operatorStack, operandStack);
        operatorStack.push_back(copy);
      }
    } else if(isParenthesesClose(copy)) {
      rearrangeStack(operatorStack, operandStack);
    } else if(isAny(copy)) {
      // ignore
    } else {
      // error
      std::cerr << "Error detecting character " << copy << std::endl;
      return false;
    }
    prevWasOperator = isOperator(copy);
    return true;
  }

  /**
   * Lexer function to detect a Function node.
   * @param valString element of DefaultFunctions vector
   * @returns
   */
  static std::shared_ptr<Function> GetFunction(const std::string& valString) {
    auto elem = DefaultFunctions.find(valString);
    return (elem != DefaultFunctions.end()) ? elem->second : nullptr;
  }

  /**
   * Recursively split of input string into function/symbol/operator/numerical substrings.
   * @param processString
   * @returns
   */
  std::vector<std::string> splitFunctionsOrElementwise(const std::string& in) {
    std::vector<std::string> out;
    auto eq = in;

    std::vector<std::string> functionNames;
    functionNames.reserve(DefaultFunctions.size());
    for(const auto& elem : DefaultFunctions) { functionNames.push_back(elem.first); }
    auto res = extractObjects(eq, functionNames);
    out.insert(out.end(), res.begin(), res.end());

    std::vector<std::string> constantNames;
    constantNames.reserve(DefaultSymbols.size());
    for(const auto& elem : DefaultSymbols) { constantNames.push_back(elem.first); }
    res = extractObjects(eq, constantNames);
    out.insert(out.end(), res.begin(), res.end());

    if(!eq.empty()) {
      auto remainingPart = split_by_regex(eq, GetRegex(MathNodeType::NodeType_Operator_or_Parentheses));
      out.insert(out.end(), remainingPart.begin(), remainingPart.end());
    }
    return out;
  }

  /**
   * Splits an equation string into a vector of strings
   *
   * @param eqString equation string to split
   * @returns vector with equation elements
   */
  std::vector<std::string> splitEquation(const std::string& eqString);

  /**
   * tests if value is number
   * @param in
   * @returns
   */
  static bool isNumber(const std::string& in) {
    static auto numberRegex = GetRegex(MathNodeType::NodeType_Numeric);
    std::cmatch m;
    return std::regex_match(in.c_str(), m, numberRegex);
  };
  /**
   * tests if value is symbol
   * @param in
   * @returns
   */
  static bool isSymbol(const std::string& in) {
    static auto symbolRegex = GetRegex(MathNodeType::NodeType_Symbolic);
    std::cmatch m;
    return std::regex_match(in.c_str(), m, symbolRegex);
  };
  /**
   * test if value is operator
   * @param in
   * @returns
   */
  static bool isOperator(const std::string& in) {
    static auto operatorRegex = GetRegex(MathNodeType::NodeType_Operator);
    std::cmatch m;
    return std::regex_match(in.c_str(), m, operatorRegex);
  };
  /**
   * test if value is anything processable
   * @param in
   * @returns
   */
  static bool isAny(const std::string& in) {
    static auto anyRegex = GetRegex(MathNodeType::NodeType_Any);
    std::cmatch m;
    return std::regex_match(in.c_str(), m, anyRegex);
  };
  /**
   * test if value is parentheses open
   * @param in
   * @returns
   */
  static bool isParenthesesOpen(const std::string& in) { return in.find('(') != std::string::npos; };
  /**
   * test if value is parentheses close
   * @param in
   * @returns
   */
  static bool isParenthesesClose(const std::string& in) { return in.find(')') != std::string::npos; };

  /**
   * process current operator until end of equation, next operator with higher priority or closing parentheses
   * @param operatorStack current operator stack object
   * @param operandStack current operand stack
   * @param currentOp the operator to process
   */
  static void processCurrentOP(
  const std::shared_ptr<Operator>& currentOp,
  std::vector<std::string>& operatorStack,
  std::vector<std::shared_ptr<MathNode>>& operandStack);

  /**
   * rearranges stack until end of equation of opening parentheses
   * @param operatorStack operator stack object
   * @param operandStack operand stack object
   */
  static void
  rearrangeStack(std::vector<std::string>& operatorStack, std::vector<std::shared_ptr<MathNode>>& operandStack);


  /**
   * extracts given objects from given equation string
   * @param eq equation string
   * @param container container containing objects to search for
   * @returns extracted objects
   */
  std::vector<std::string> extractObjects(std::string& eq, const std::vector<std::string>& container);
};
/**
 * \example symb/TestSymbolic.cpp
 * This is an example on how to use the symb module.
 */