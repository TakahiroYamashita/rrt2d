#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>


namespace ob = ompl::base;
namespace og = ompl::geometric;

// Return true if the state is valid, false if the state is invalid
bool isStateValid(const ob::State *state)
{
  const ob::RealVectorStateSpace::StateType* state2D=state->as<ob::RealVectorStateSpace::StateType>();

  // Extract the robot's (x,y) position from its state
  double x = state2D->values[0];
  double y = state2D->values[1];

  
  if(x>=-0.5 && x<=0.8 && y<=1.0 && y>=0.2)
    return false;

  if(x>=-0.2 && x<=0.7 && y<=0.2 && y>=-0.5)
    return false;

  if(x>=-1.0 && x<=-0.5 && y<=-0.2 && y>=-0.8)
    return false;

  // Otherwise, the state is valid:
  return true;
}


bool planWithSimpleSetup(void)
{
  bool found_path=false;
  std::random_device rdev;
  std::mt19937 engine(rdev());

  std::uniform_real_distribution<> dist(0.5, 1.0);
  std::uniform_real_distribution<> dists(0, 0.5);
  
  float start_x=0, start_y=0;
  float goal_x=0, goal_y=0;

  start_x = -1*dist(engine);
  start_y = dist(engine);

  goal_x = dist(engine);
  goal_y = -1*dist(engine);

  
  clock_t starts = clock();
  

  // Construct the state space where we are planning
  ob::StateSpacePtr space(new ob::RealVectorStateSpace(2));

  ob::RealVectorBounds bounds(2);
  bounds.setLow(-1);
  bounds.setHigh(1);
  space->as<ob::RealVectorStateSpace>()->setBounds(bounds);

  // Instantiate SimpleSetup
  og::SimpleSetup ss(space);

  // Setup the StateValidityChecker
  ss.setStateValidityChecker(boost::bind(&isStateValid, _1));

  std::cout << "tout va bien"<<std::endl;
  // Setup Start and Goal

  ob::ScopedState<> start(space);
  start->as<ob::RealVectorStateSpace::StateType>()->values[0] = start_x;
  start->as<ob::RealVectorStateSpace::StateType>()->values[1] = start_y;

  ob::ScopedState<> goal(space);
  goal->as<ob::RealVectorStateSpace::StateType>()->values[0] = goal_x;
  goal->as<ob::RealVectorStateSpace::StateType>()->values[1] = goal_y;

  ss.setStartAndGoalStates(start, goal);

  // Execute the planning algorithm
  ob::PlannerStatus solved = ss.solve(1.0);

  
  if (solved)
  {
    srand(time(NULL));
    // Simplify the solution 
    ss.simplifySolution();
    
    // Print the solution path to a file
    std::ofstream ofs("../plot/path.dat", std::ios::app);
    ss.getSolutionPath().printAsMatrix(ofs);

  }
  else {std::cout << "No solution found" << std::endl;}

  clock_t end = clock();     // 終了時間

  double rrttime = end - starts;
  std::cout << "time = " << rrttime / CLOCKS_PER_SEC << "sec.\n";
}




int main()
{
  std::ofstream ofs("../plot/path.dat", std::ios::trunc);
  int i;
  for(i = 0; i<1000; i++)
    planWithSimpleSetup();
  return 0;
  }


