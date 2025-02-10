#pragma once

#include <ctime>
#include <random>

static bool TimeInitialized = false;
static int RandomSeed = 2;
static std::mt19937 rng; // Mersenne Twister engine

/**
 * RNG
 */
class Random
{
  /**
   * Initialize and set seed
   * @param useSeed
   */
  static void InitTime(bool useSeed = true) {
    rng.seed(useSeed ? RandomSeed : static_cast<int>(time(0)));
    TimeInitialized = true;
  }

public:
  /**
   * Sets seed to given value
   * @param seed
   */
  static void SetSeed(int seed) {
    RandomSeed = seed;
    TimeInitialized = false;
  }

  /**
   * Getter for random number between (including borders) l and r
   * @param l
   * @param r
   * @returns
   */
  static double Get(double l = 0.0, double r = 1.0) {
    if (!TimeInitialized) InitTime(true);

    std::uniform_real_distribution<> dist(l, r);
    return dist(rng);
  }
};
/**
 * \example TestRandom.cpp
 * This is an example on how to use the Random class.
 */
