#pragma once
#include "../format.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <regex>
#include <string>

/**
 * Node type representation
 */
enum MathNodeType {
  NodeType_Operator                = 0,
  NodeType_Symbolic                = 1,
  NodeType_Numeric                 = 2,
  NodeType_Parentheses             = 3,
  NodeType_Functional              = 4,
  NodeType_DefaultSymbol           = 5,
  NodeType_Any                     = 6,
  NodeType_Operator_or_Parentheses = 7,
};

enum NodeConnectionType {
  ConnectionType_Dual    = 0,
  ConnectionType_Left    = 2,
  ConnectionType_Right   = 3,
  ConnectionType_None    = 4,
  ConnectionType_Unknown = 5,
};

/**
 * regex factory for node types
 * @param type
 * @returns regex for given type
 */
std::regex GetRegex(MathNodeType type);


/**
 * Base node type of Abstract syntax tree implementation
 */
class MathNode
{
public:
  //! helper to determine whether a node is negated or not
  bool isNegative = false;
  //! helper to determine if node is within parentheses
  bool hasParentheses = false;
  //! node type representation
  MathNodeType type;
  //! type of connection allowed for this node
  NodeConnectionType connectionType = NodeConnectionType::ConnectionType_Dual;
  //! next node on left side
  std::shared_ptr<MathNode> left = nullptr;
  //! next node on right side
  std::shared_ptr<MathNode> right = nullptr;
  //! char representation of value
  char* value{};

protected:
  //! helper to store size of value
  size_t valSize = 0;

public:
  /**
   * Default constructor
   * @param val
   */
  explicit MathNode(const std::string& val) {
    auto realValue = strip(val); // strip to allow comparison
    isNegative     = realValue.find('-') != std::string::npos && realValue.size() > 1;
    value          = new char[realValue.size() + 1];
    std::copy(realValue.begin(), realValue.end(), value);
    value[realValue.size()] = '\0'; // don't forget the terminating 0
    valSize                 = realValue.size() + 1;
  }

  /**
   * Copy constructor
   * @param other
   */
  MathNode(const MathNode& other) {
    valSize        = other.valSize;
    isNegative     = other.isNegative;
    hasParentheses = other.hasParentheses;
    value          = (char*)malloc(other.valSize);
    strcpy(value, other.value);
    left  = other.left;
    right = other.right;
    type  = other.type;
  }

  /**
   * Interface function for evaluation of node,
   * needs to be implemented to add new type of node
   * @returns
   */
  [[nodiscard]] virtual double Evaluate() const = 0;

  /**
   * Getter for string representation of node
   * @returns
   */
  [[nodiscard]] std::string GetString() const {
    std::stringstream out;
    if(type == NodeType_Parentheses || hasParentheses) out << "(";
    if(left != nullptr) out << left;
    out << value;
    if(right != nullptr) out << right;
    if(type == NodeType_Parentheses || hasParentheses) out << ")";
    return out.str();
  }

  friend bool operator==(const MathNode& lhs, const MathNode& rhs) {
    return strcmp(lhs.value, rhs.value) == 0 && lhs.isNegative == rhs.isNegative;
  }

  bool operator==(MathNode* rhs) { return strcmp(value, rhs->value) == 0 && isNegative == rhs->isNegative; }

  /**
   * beautified output operator
   * @param ostr
   * @param other
   * @returns
   */
  friend std::ostream& operator<<(std::ostream& ostr, const MathNode& other) {
    if(other.hasParentheses) ostr << "(";
    if(other.left != nullptr) ostr << other.left;
    ostr << (other.isNegative ? "-" : "") << other.value;
    if(other.right != nullptr) ostr << other.right;
    if(other.hasParentheses) ostr << ")";
    return ostr;
  }
  /**
   * beautified output operator
   * @param ostr
   * @param other
   * @returns
   */
  friend std::ostream& operator<<(std::ostream& ostr, MathNode* other) {
    // early exit
    if(other == nullptr) return ostr;

    if(other->hasParentheses) ostr << "(";
    if(other->left != nullptr) ostr << other->left;
    ostr << (other->isNegative ? "-" : "") << other->value;
    if(other->right != nullptr) ostr << other->right;
    if(other->hasParentheses) ostr << ")";
    return ostr;
  }
};

/**
 * \example symb/TestSymbolic.cpp
 * This is an example on how to use the symb module.
 */
