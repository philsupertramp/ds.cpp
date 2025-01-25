#include "../Test.h"

class SGDTestCase : public Test
{
  bool TestUpdateWeights() { return true; }

public:
  virtual void run() { }
};

int main() {
  SGDTestCase().run();
  return 0;
};
