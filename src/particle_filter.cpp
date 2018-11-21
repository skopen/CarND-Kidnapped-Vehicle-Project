/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <float.h>
#include <map>
#include <random>


#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// start with a set of particles
	num_particles = 200;

	// Create Gaussian distribution for x, y, theta
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Create necessary particles, and assign them ids, positions and heading (randomly per Gaussian distribution)
	for (int i = 0; i < num_particles; i++)
	{
		Particle part;

		part.id = i;
		part.x = dist_x(gen);
		part.y = dist_y(gen);
		part.theta = dist_theta(gen);

		part.weight = 1.0;

		particles.push_back(part);
		weights.push_back(1.0);
	}

	// set initialized to true
	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/0


	// set up Gaussians
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	// predict new location and heading based on velocity, yaw rate and elapsed time
	for (int i = 0; i < num_particles; i++)
	{
		// accommodate for divide by zero by downcasting to a simpler yaw free model
		if (fabs(yaw_rate) < 0.000001)
		{
			particles[i].x += velocity*delta_t*cos(particles[i].theta) + dist_x(gen);
			particles[i].y += velocity*delta_t*sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
		else
		{
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + dist_y(gen);
			particles[i].theta += yaw_rate*delta_t + dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// do the O(m*n) processing needed to associate observations with landmarks based on proximity
	for (unsigned int i = 0; i < observations.size(); i++)
	{
		double minDist = DBL_MAX;
		int landmark_idx = -1;

		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (d < minDist)
			{
				minDist = d;
				landmark_idx = j;
			}
		}

		observations[i].id = predicted[landmark_idx].id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// get standard deviations for landmarks
	double sx = std_landmark[0];
	double sy = std_landmark[1];

	// run through all particles
	for (int i = 0; i < num_particles; i++)
	{
		Particle p = particles[i];

		double prob = 1.0;

		// run through all observations
		for (unsigned int j = 0; j < observations.size(); j++)
		{
			LandmarkObs obs = observations[j];

			double xm = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
			double ym = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;

			double minDist = DBL_MAX;
			int landmark_idx = -1;

			// identify the landmark associated with this observation to get mux and muy (not using the dataAssociation()
			// method, but that is also an option
			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++)
			{
				double distance = dist(xm, ym, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
				if (distance < minDist)
				{
					minDist = distance;
					landmark_idx = k;
				}
			}

			double mux = map_landmarks.landmark_list[landmark_idx].x_f;
			double muy = map_landmarks.landmark_list[landmark_idx].y_f;

			// find the probability (weight) that this obersation is possible for the particle in consideration
			double pxy = (1.0/(2*M_PI*sx*sy))*exp(-((((xm - mux)*(xm - mux))/(2*sx*sx)) + (((ym - muy)*(ym - muy))/(2*sy*sy))));

			// calculate effective probability for fully independent events/observations
			prob *= pxy;
		}

		// update weights as a proxy for likelihood/probability
		particles[i].weight = prob;
		weights[i] = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// use standard discrete distribution model
    std::discrete_distribution<> d(weights.begin(), weights.end());

    std::vector<Particle> newParticles;

    // resample based on weights
    for(unsigned int i = 0; i < particles.size(); i++)
    {
        newParticles.push_back(particles[d(gen)]);
    }

    // swap out old particles for newly sampled ones
    particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
