#pragma once
#include "MyVars.h"
#include <vector>

using namespace std;
class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	~Neuron();
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	const vector<Connection> getOutputWeights() { return m_outputWeights; };
	void setOutputWeights(vector<Connection> connection) { m_outputWeights = connection;  };

	void setEta(double _eta);
	void setAlpha(double _alpha);
	void setFunc(activationFunc _func);

private:
	double eta; // [0.0..1.0] overall net training rate
	double alpha; // [0.0..n] multiplier of last weight change (momentum)
	double transferFunction(double x);
	double transferFunctionDerivative(double x);
	activationFunc m_transferFunc;
	
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};


