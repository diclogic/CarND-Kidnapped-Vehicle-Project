/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using namespace std;

random_device rd{};
default_random_engine gen{rd()};

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50; // TODO: Set the number of particles

  weights.resize(num_particles, 0);
  particles.resize(num_particles);

  std::normal_distribution<> dist_x{x, std[0]};
  std::normal_distribution<> dist_y{y, std[1]};
  std::normal_distribution<> dist_theta{theta, std[2]};

  for (int n = 0; n < num_particles; ++n)
  {
    Particle p;
    p.id = n;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.;
    particles[n] = move(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  using norm_dist = std::normal_distribution<double>;

  auto dt = delta_t;
  auto v = velocity;
  auto yr = yaw_rate;
  auto dtheta = yr * dt;

  auto &std = std_pos;
  norm_dist noise_dist[] = {norm_dist{0, std[0]}, norm_dist{0, std[1]}, norm_dist{0, std[2]}};

  for (auto &p : particles)
  {
    // bicycle motion model
    if (yr == 0)
    {
      p.x += v * dt * cos(p.theta);
      p.y += v * dt * sin(p.theta);
    }
    else
    {
      p.x += v * (sin(p.theta + dtheta) - sin(p.theta)) / yr;
      p.y += v * (cos(p.theta) - cos(p.theta + dtheta)) / yr;
      p.theta += dtheta;
    }

    // add noise
    p.x += noise_dist[0](gen);
    p.y += noise_dist[1](gen);
    p.theta += noise_dist[2](gen);
  }
}

vector<LandmarkObs> convertToMapCoords(const vector<LandmarkObs> &obs_list, const Particle &p)
{
  size_t size = obs_list.size();

  double cos_theta = cos(p.theta);
  double sin_theta = sin(p.theta);

  vector<LandmarkObs> retval(size);
  for (int i = 0; i < size; ++i)
  {
    auto &obs = obs_list[i];
    retval[i].x = p.x + cos_theta * obs.x - sin_theta * obs.y;
    retval[i].y = p.y + sin_theta * obs.x + cos_theta * obs.y;
  }
  return retval;
}

template <typename T>
double distance_sq(double x, double y, const T &ptObs)
{
  double dx = ptObs.x - x;
  double dy = ptObs.y - y;
  return dx * dx + dy * dy;
}

int findClosestLandmark(const LandmarkObs &obs, const vector<LandmarkObs> &landmarks, double max_range)
{
  double min_dist_sq = std::numeric_limits<double>::max();
  int minId = -1;
  for (const auto &pt : landmarks)
  {
    double dist_sq = distance_sq(obs.x, obs.y, pt);
    if (dist_sq < min_dist_sq && dist_sq < (max_range * max_range))
    {
      min_dist_sq = dist_sq;
      minId = pt.id;
    }
  }

  return minId;
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs> &all_inrange, // all landmarks in range of one particle
                                     vector<LandmarkObs> &observations, double max_range)
{
  /**
	 * TODO: Find the predicted measurement that is closest to each
	 *   observed measurement and assign the observed measurement to this
	 *   particular landmark.
	 * NOTE: this method will NOT be called by the grading code. But you will
	 *   probably find it useful to implement this method and use it as a helper
	 *   during the updateWeights phase.
	 */

  for (auto &ob : observations)
  {
    int lmId = findClosestLandmark(ob, all_inrange, max_range);
    ob.id = lmId;
  }
}

struct MapBBox
{
  double min_x;
  double min_y;
  double max_x;
  double max_y;
  double width() const { return max_x - min_x; }
  double height() const { return max_y - min_y; }
};

struct Point
{
  double x;
  double y;
};

MapBBox calcMapSize(const Map &map)
{
  double min_x = 10000, min_y = 10000, max_x = 0, max_y = 0;
  for (auto &pt : map.landmark_list)
  {
    if (pt.x_f < min_x)
      min_x = pt.x_f;
    if (pt.x_f > max_x)
      max_x = pt.x_f;
    if (pt.y_f < min_y)
      min_y = pt.y_f;
    if (pt.y_f > max_y)
      max_y = pt.y_f;
  }

  return MapBBox{min_x, min_y, max_x, max_y};
}

const vector<LandmarkObs> &cropMap(const Map &map, const Particle &p, double max_range)
{
  static const double gridSize = max_range / 2;
  static const double cache_range = max_range * 1.26;
  static const MapBBox mapSize = calcMapSize(map);
  static vector<const vector<LandmarkObs> *> gridCache = vector<const vector<LandmarkObs> *>((int)ceil(mapSize.width() / gridSize) * (int)ceil(mapSize.height() / gridSize));

  //vector<LandmarkObs> retval;
  //retval.reserve(5);

  int grid_x = (int)floor((p.x - mapSize.min_x) / gridSize);
  int grid_y = (int)floor((p.y - mapSize.min_y) / gridSize);
  int gridId = grid_x + grid_y * (int)ceil(mapSize.width() / gridSize);

  if (gridCache[gridId])
    return *gridCache[gridId];

  auto *pRet = new vector<LandmarkObs>();
  pRet->reserve(20);

  double range_sq = cache_range * cache_range;
  Point pt;
  pt.x = grid_x * gridSize + .5 * gridSize + mapSize.min_x;
  pt.y = grid_y * gridSize + .5 * gridSize + mapSize.min_y;
  for (const auto &lm : map.landmark_list)
  {
    if (distance_sq(lm.x_f, lm.y_f, pt) < range_sq)
      pRet->emplace_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
  }

  gridCache[gridId] = pRet;
  return *pRet;
}

double ParticleFilter::calcWeight(const Map &map_landmarks, double norm, std::vector<LandmarkObs, std::allocator<LandmarkObs>> &obs, double variance_x_inv, double variance_y_inv)
{

  double w_final = 1.;
  for (int i = 0; i < obs.size(); ++i)
  {
    auto &lm = obs[i];
    double x_lm = map_landmarks.landmark_list[lm.id - 1].x_f;
    double y_lm = map_landmarks.landmark_list[lm.id - 1].y_f;
    double err_x = lm.x - x_lm;
    double err_y = lm.y - y_lm;
    double exponent = err_x * err_x * variance_x_inv + err_y * err_y * variance_y_inv;
    double w = norm * exp(-exponent);
    w_final *= w;
  }
  return w_final;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double norm = 1. / (2. * M_PI * std_landmark[0] * std_landmark[1]);
  double variance_x_inv = 1. / (2. * std_landmark[0] * std_landmark[0]);
  double variance_y_inv = 1. / (2. * std_landmark[1] * std_landmark[1]);

  vector<int> ids(observations.size());
  vector<double> xs(observations.size());
  vector<double> ys(observations.size());

  int size = particles.size();
  for (int j = 0; j < size; ++j)
  {
    auto &p = particles[j];
    auto obs = convertToMapCoords(observations, p);
    auto &inrange_lms = cropMap(map_landmarks, p, sensor_range);
    dataAssociation(inrange_lms, obs, sensor_range);
    for (int k = 0; k < obs.size(); ++k)
    {
      ids[k] = obs[k].id;
      xs[k] = obs[k].x;
      ys[k] = obs[k].y;
      SetAssociations(p, ids, xs, ys);
    }

    double w_final = calcWeight(map_landmarks, norm, obs, variance_x_inv, variance_y_inv);

    p.weight = w_final;
    weights[j] = w_final;
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  size_t count = particles.size();
  vector<Particle> new_set;
  new_set.reserve(count);

  //*
  std::discrete_distribution<int> distrib(weights.begin(), weights.end());

  for (int i = 0; i < particles.size(); ++i)
  {
    int j = distrib(gen);
    new_set.push_back(particles[j]);
    new_set.back().id = i;
  }
  // */

  /*
  double w_max = accumulate(weights.begin(), weights.end(), -1., [](double l,double r){ return l>r?l:r; });
  std::uniform_real_distribution<> distrib(0.,2.*w_max);
	
  int idx = gen()%count;
  double beta = 0.;
  for (int i=0; i<count; ++i)
  {
    beta += distrib(gen);
    
    while (weights[idx] < beta)
    {
      beta -= weights[idx];
      idx = (idx+1)%count;
    }
    new_set.push_back(particles[idx]);
  }
  */

  particles.swap(new_set);
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
