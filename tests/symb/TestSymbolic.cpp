#include "../Test.h"
#include <math/symb/Equation.h>


class SymbolicTestCase : public Test
{
  friend Equation;

  bool TestParser() {
    Equation eq("x+2-((3*5)-x)-2");

    return true;
  }

  bool TestMathematicalProperties() {
    Equation equation = Equation("2 + x + 1 + 2 + y");
    AssertEqual(equation(1, 2), 8);
    AssertEqual(equation(3, 3), 11);
    AssertEqual(equation(6, 4), 15);

    equation = Equation("x * 2");
    AssertEqual(equation(1), 2);
    AssertEqual(equation(3), 6);
    AssertEqual(equation(6), 12);

    equation = Equation("5 + x * 2");
    AssertEqual(equation(1), 7);
    AssertEqual(equation(3), 11);
    AssertEqual(equation(6), 17);

    Equation equation1("x * 2 + 5");
    for(int i = -10; i < 10; ++i) { AssertEqual(equation(i), equation1(i)); }
    equation  = Equation("5 * 3 + 2 * x");
    equation1 = Equation("5 * 3 + x * 2");
    Equation equation2("2 * x + 5 * 3");
    Equation equation3("x * 2 + 3 * 5");
    for(int i = -10; i < 10; ++i) {
      // Commutative property of multiplication
      AssertEqual(equation(i), equation1(i));
      // Commutative property of addition
      AssertEqual(equation1(i), equation2(i));
      // Commutative property of addition and multiplication
      AssertEqual(equation2(i), equation3(i));
    }

    equation = Equation("5 / x  - 2");
    AssertEqual(equation(1), 3);
    AssertEqual(equation(5), -1);
    AssertEqual(equation(2), 0.5);

    equation = Equation("x^2  - 5");
    AssertEqual(equation(1), -4);
    AssertEqual(equation(5), 20);
    AssertEqual(equation(2), -1);

    return true;
  }
  bool TestNegative() {
    Equation equation("-x + 5");
    AssertEqual(equation(2), 3);
    AssertEqual(equation(-2), 7);
    equation = Equation("-x + -5");
    AssertEqual(equation(2), -7);
    AssertEqual(equation(-2), -3);
    equation = Equation("-x - -5");
    AssertEqual(equation(2), 3);
    AssertEqual(equation(-2), 7);
    return true;
  }

  bool TestPower() {
    Equation eq("2 + x^2");

    AssertEqual(eq(2), 6);

    eq = Equation("2 * x^2");
    AssertEqual(eq(2), 8);
    eq = Equation("2 * x^(3+y)");
    AssertEqual(eq(2, 1), 2 * (2 * 2 * 2 * 2));
    eq = Equation("2 * x ^ (3+y)");
    AssertEqual(eq(2, 1), 2 * (2 * 2 * 2 * 2));
    return true;
  }

  bool TestParentheses() {
    Equation equation("2 * (x - 1)");
    AssertEqual(equation(1), 0);
    AssertEqual(equation(2), 2);
    AssertEqual(equation(60), 118);
    equation = Equation("2 * (x - (1 * y)) - 2");

    AssertEqual(equation(1, 1), -2);
    AssertEqual(equation(1, 2), -4);

    equation = Equation("2 * (x - 1 + y) - 2");

    AssertEqual(equation(1, 1), 0);
    AssertEqual(equation(1, 2), 2);

    equation = Equation("(x - 1 + y) * 2 - 2");

    AssertEqual(equation(1, 1), 0);
    AssertEqual(equation(1, 2), 2);

    equation = Equation("2 * (x - 1) - 2");

    AssertEqual(equation(1), -2);
    AssertEqual(equation(2), 0);
    AssertEqual(equation(3), 2);
    AssertEqual(equation(99), 194);
    return true;
  }

  bool TestFailsGracefully() {
    Equation eq("µ");
    AssertTrue(eq.baseNode == nullptr);
    return true;
  }

  bool TestGetRegex() {
    auto sym = GetRegex(MathNodeType::NodeType_Symbolic);
    auto num = GetRegex(MathNodeType::NodeType_Numeric);
    auto ope = GetRegex(MathNodeType::NodeType_Operator);
    auto par = GetRegex(MathNodeType::NodeType_Parentheses);
    auto any = GetRegex(MathNodeType::NodeType_Any);

    std::cmatch m;
    AssertTrue(std::regex_match("x", m, sym));
    AssertFalse(std::regex_match("+", m, sym));

    AssertTrue(std::regex_match("2", m, num));
    AssertFalse(std::regex_match("x", m, num));

    AssertTrue(std::regex_match("+", m, ope));
    AssertFalse(std::regex_match("2", m, ope));

    AssertTrue(std::regex_match("(1)", m, par));
    AssertFalse(std::regex_match("1", m, par));

    AssertTrue(std::regex_match("^", m, any));
    AssertFalse(std::regex_match("µ", m, any));

    return true;
  }
  bool TestGenerateOperator() {
    const std::shared_ptr<Operator>& pow       = GenerateOperator(OperatorType::TYPE_POWER);
    const std::shared_ptr<Operator>& add       = GenerateOperator(OperatorType::TYPE_ADDITION);
    const std::shared_ptr<Operator>& sub       = GenerateOperator(OperatorType::TYPE_SUBTRACTION);
    const std::shared_ptr<Operator>& mul       = GenerateOperator(OperatorType::TYPE_MULTIPLICATION);
    const std::shared_ptr<Operator>& div       = GenerateOperator(OperatorType::TYPE_DIVISION);
    const std::shared_ptr<Operator>& par_open  = GenerateOperator(OperatorType::TYPE_PARENTHESES_OPEN);
    const std::shared_ptr<Operator>& par_close = GenerateOperator(OperatorType::TYPE_PARENTHESES_CLOSE);
    const std::shared_ptr<Operator>& empty     = GenerateOperator(static_cast<OperatorType>(-1));

    AssertEqual(pow->value, "^");
    AssertEqual(pow->op(1, 2), 1);

    AssertEqual(add->value, "+");
    AssertEqual(add->op(1, 2), 3);

    AssertEqual(sub->value, "-");
    AssertEqual(sub->op(1, 2), -1);

    AssertEqual(mul->value, "*");
    AssertEqual(mul->op(1, 2), 2);

    AssertEqual(div->value, "/");
    AssertEqual(div->op(1, 2), 0.5);

    AssertEqual(par_open->value, "(");
    AssertEqual(par_open->op(1, 2), 0.0);

    AssertEqual(par_close->value, ")");
    AssertEqual(par_close->op(1, 2), 0.0);

    AssertEqual(empty->value, "");
    AssertEqual(empty->op(1, 2), -1);

    return true;
  }

  bool TestChaining() {
    Equation a("x");
    Equation b("1");

    auto combined = Equation::Chain(a, b, GenerateOperator(OperatorType::TYPE_ADDITION));

    Equation expected("x + 1");
    for(int i = -10; i < 10; ++i) AssertEqual(expected(i), combined(i));

    // can chain using different operator
    a        = Equation("x");
    b        = Equation("1");
    combined = Equation::Chain(a, b, GenerateOperator(OperatorType::TYPE_SUBTRACTION));
    expected = Equation("x - 1");

    for(int i = -10; i < 10; ++i) AssertEqual(combined(i), expected(i));

    // can chain using same symbols
    a        = Equation("x");
    b        = Equation("x");
    combined = Equation::Chain(a, b, GenerateOperator(OperatorType::TYPE_ADDITION));
    expected = Equation("x + x"); // although real equation is "x + x"

    for(int i = -10; i < 10; ++i) AssertEqual(combined(i), expected(i));

    // can chain using different symbols
    a        = Equation("x");
    b        = Equation("y");
    combined = Equation::Chain(a, b, GenerateOperator(OperatorType::TYPE_ADDITION));
    expected = Equation("x + y");

    for(int i = -10; i < 10; ++i) AssertEqual(combined(1, i), expected(1, i));
    return true;
  }

  bool TestFunctions() {
    Equation equation("sqrt(x)");
    AssertEqual(equation(1.0), 1.0);
    AssertEqual(equation(2.0), 1.414213562373095);
    equation = Equation("sqrt(x) + 1");
    AssertEqual(equation(1.0), 2.0);
    AssertEqual(equation(2.0), 2.414213562373095);
    equation = Equation("2 * 3 + sqrt(x) + 1");
    AssertEqual(equation(1.0), 8.0);
    AssertEqual(equation(2.0), 8.414213562373095);
    equation = Equation("2 * 3 + sqrt(x+1) + 1");
    AssertEqual(equation(0.0), 8.0);
    AssertEqual(equation(1.0), 8.414213562373095);
    equation = Equation("2*3+sqrt(x+1) + 1");
    AssertEqual(equation(0.0), 8.0);
    AssertEqual(equation(1.0), 8.414213562373095);

    equation = Equation("sqrt(x)^2");
    AssertEqual(equation(1.0), 1.0);
    AssertEqual(equation(2.0), 2.0);

    equation = Equation("sin(x)");
    AssertEqual(equation(0), 0);
    AssertEqual(equation(1), 0.8414709848078965);

    equation = Equation("sin(pi)");
    AssertEqual(equation(), 0);
    equation = Equation("cos(pi)");
    AssertEqual(equation(), -1);
    equation = Equation("log(1)");
    AssertEqual(equation(), 0);

    Equation circle("r^2 * pi");

    AssertEqual(circle(1), 3.1415926535897932384626433832795028841971693993751058209749445923078164);

    return true;
  }

  bool TestDegree() {
    Equation eq("x^2");
    AssertEqual(eq.degree, 2);
    eq = Equation("x^1");
    AssertEqual(eq.degree, 1);
    eq = Equation("x^10");
    AssertEqual(eq.degree, 10);
    eq = Equation("3*x^3+5*y^5");
    AssertEqual(eq.degree, 5);
    return true;
  }

  bool TestGetString() {
    Equation eq("x");
    std::cout << eq.GetString() << std::endl;
    AssertEqual(eq.GetString(), "x");

    eq = Equation("x^2");
    std::cout << eq.GetString() << std::endl;
    AssertEqual(eq.GetString(), "x^2.000000");

    eq = Equation("x+1");
    std::cout << eq.GetString() << std::endl;
    AssertEqual(eq.GetString(), "x+1.000000");

    eq = Equation("x*(y-1)^2");
    // TODO: this is a false notation.
    AssertEqual(eq.GetString(), "x*y+-1.000000^2.000000");

    return true;
  }

  bool TestSimplify() {
    // x-11
    Equation eq("1*3 + x - 5 * 3 - 2 + 3");
    std::string expected = "((3.000000+x)+-14.000000)\n";
    std::stringstream stream;
    eq.Simplify();
    eq.Print(stream);

    AssertEqual(expected, stream.str());

    eq = Equation("1*3 + x - 5 * 3 - 2 + 3");
    eq.Simplify();

    AssertEqual(Equation("1*3 + x - 5 * 3 - 2 + 3")(5), -6);
    AssertEqual(eq(5), -6);
    return true;
  }

  bool TestDefaultSymbols() { return true; }

public:
  virtual void run() {
    TestPower();
    TestMathematicalProperties();
    TestNegative();
    TestParentheses();
    TestFailsGracefully();
    TestGetRegex();
    TestGenerateOperator();
    TestChaining();
    TestFunctions();
    TestDefaultSymbols();
    TestSimplify();
    TestParser();
    TestDegree();
    TestGetString();
  }
};

int main() {
  SymbolicTestCase().run();
  return 0;
}
