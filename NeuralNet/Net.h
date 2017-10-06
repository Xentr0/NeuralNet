#pragma once
#include "Neuron.h"
#include "MyVars.h"
#include <string>
#include <vector>

using namespace std;

//class Neuron;
typedef vector<Neuron> Layer;

class Net
{
public:	
	Net(const Params _params);
	~Net();
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	void writeWeights(string path);
	void copyWeights(const vector<double> &weights);

private:
	vector<Layer> m_layers; // m_layers [layerNum][neuronNum]
	double m_error;
	Params m_params;
};

