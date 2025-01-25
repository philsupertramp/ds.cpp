#pragma once

#include <ctime>
#include <random>

static bool TimeInitialized = false;
static int RandomSeed       = 2;
/**
 * RNG
 */
class Random
{
  /**
   * initialize and set seed
   * @param useSeed
   */
  static void InitTime(bool useSeed = true) {
    std::srand(useSeed ? RandomSeed : (int)time(0));
    TimeInitialized = true;
  }

public:
  /**
   * sets seed to given value
   * @param seed
   */
  static void SetSeed(int seed) {
    RandomSeed      = seed;
    TimeInitialized = false;
  }
  /**
   * Getter for random number between (including borders) l and r
   * @param l
   * @param r
   * @returns
   */
  static double Get(double l = 0.0, double r = 1.0) {
    if(!TimeInitialized) InitTime(false);

    std::random_device rd;                       // create object for seeding
    std::mt19937 gen{ rd() };                    // create engine and seed it
    std::uniform_real_distribution<> dist(l, r); // create distribution for integers with [1; 9] range
    return dist(gen);
  }
};
/**
 * \example TestRandom.cpp
 * This is an example on how to use the Random class.
 */
