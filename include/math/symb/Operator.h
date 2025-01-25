#pragma once
#include "MathNode.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_set>


/**
 * Priority definition for AST generation, used in shunting-yard algorithm,
 * the Equation class builds nodes using a stack based on priorities
 */
enum OperatorPriority {
  OPClassUnknown     = 0,
  OPClassLine        = 1,
  OPClassDot         = 2,
  OPClassPow         = 3,
  OPClassParentheses = 4,
  OPClassFunction    = 5,
};

/**
 * Representation of operator type used in factories
 */
enum OperatorType {
  TYPE_ADDITION          = 0, // ConnectionType_Dual
  TYPE_SUBTRACTION       = 1, // ConnectionType_Dual
  TYPE_MULTIPLICATION    = 2, // ConnectionType_Dual
  TYPE_DIVISION          = 3, // ConnectionType_Dual
  TYPE_POWER             = 4, // ConnectionType_Dual
  TYPE_PARENTHESES_OPEN  = 5, // ConnectionType_None
  TYPE_PARENTHESES_CLOSE = 6, // ConnectionType_None
};

/**
 * representation of operator value in char
 */
enum OperatorValue : char {
  //! l + r
  VALUE_ADDITION = '+',
  //! l - r
  VALUE_SUBTRACTION = '-',
  //! l * r
  VALUE_MULTIPLICATION = '*',
  // l / r
  VALUE_DIVISION = '/',
  //! l^r
  VALUE_POWER = '^',
  //! (
  VALUE_PARENTHESES_OPEN = '(',
  //! )
  VALUE_PARENTHESES_CLOSE = ')',
};

/**
 * Left-to-right operator representation
 */
class Operator : public MathNode
{
public:
  //! Operator implementation, this will be evaluated during `Evaluate` calls
  std::function<double(double, double)> op;
  //! Operator priority, used to create AST
  OperatorPriority priority;
  /**
   * default constructor
   * @param name
   * @param fun
   * @param operatorPriority
   */
  Operator(const std::string& name, std::function<double(double, double)> fun, OperatorPriority operatorPriority)
    : MathNode(name)
    , op(std::move(fun))
    , priority(operatorPriority) {
    type = MathNodeType::NodeType_Operator;
  }

  /**
   * copy constructor
   * @param other
   */
  Operator(const Operator& other)
    : MathNode(other) {
    op       = other.op;
    priority = other.priority;
  }

  /**
   * evaluates op(left, right)
   * @returns
   */
  [[nodiscard]] double Evaluate() const override {
    if(priority == OPClassParentheses) { return left->Evaluate(); }
    assert(left != nullptr && right != nullptr);
    return op(left->Evaluate(), right->Evaluate());
  }
};

/**
 * Operator factory, returns shared_ptr to operator with given type
 * @param type
 * @returns
 */
std::shared_ptr<Operator> GenerateOperator(OperatorType type);


/**
 * Function representation as node.
 *
 * Current implementation represents single operand functions, like
 * \f[
 *  \sqrt{x} \quad \text{ or } \quad \log{x}
 * \f]
 */
class Function : public Operator
{
public:
  /**
   * Single operand default constructor
   * @param val string representation
   * @param fun single value evaluation function
   */
  Function(const std::string& val, const std::function<double(double)>& fun)
    : Operator(
    val, [fun](double a, [[maybe_unused]] double b) { return fun(a); }, OperatorPriority::OPClassFunction) {
    connectionType = NodeConnectionType::ConnectionType_Left;
    type           = MathNodeType::NodeType_Functional;
  }

  /**
   * Two operand default constructor
   * @param val string representation
   * @param fun two value evaluation function
   *
   * TODO: enable parsing for this kind of function!
   */
  Function(const std::string& val, const std::function<double(double, double)>& fun)
    : Operator(val, fun, OperatorPriority::OPClassFunction) {
    connectionType = NodeConnectionType::ConnectionType_Dual;
    type           = MathNodeType::NodeType_Functional;
    std::cerr << "This can't be parsed yet!!!!!!" << std::endl;
  }

  [[nodiscard]] double Evaluate() const override {
    if(connectionType == NodeConnectionType::ConnectionType_Left) return op(left->Evaluate(), 0.0);
    /**
     * TODO: whenever enabled comment out.
    else if(connectionType == NodeConnectionType::ConnectionType_Dual)
        return op(left->Evaluate(), right->Evaluate());
    **/
    return NAN;
  }

  // TODO: add beautified string representation
};
/**
 * Default functions for parser
 */
const std::map<std::string, std::shared_ptr<Function>> DefaultFunctions = {
  { "sqrt", std::make_shared<Function>("sqrt", [](double a) { return sqrt(a); }) },
  { "log", std::make_shared<Function>("log", [](double a) { return log(a); }) },
  { "sin", std::make_shared<Function>("sin", [](double a) { return sin(a); }) },
  { "cos", std::make_shared<Function>("cos", [](double a) { return cos(a); }) },
};
/**
 * Helper function to test if string is function representation
 * @param in
 * @returns
 */
bool isFunction(const std::string& in);

/**
 * \example symb/TestSymbolic.cpp
 * This is an example on how to use the symb module.
 */