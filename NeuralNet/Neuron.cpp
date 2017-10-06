#include "Neuron.h"
//#include <cmath>

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}


Neuron::~Neuron()
{
}

void Neuron::feedForward(Layer & prevLayer)
{
	double sum = 0.0;
	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);					
}

void Neuron::updateInputWeights(Layer & prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			// Individual input, magnified by the gradient and train rate:
			eta
			* neuron.getOutputVal()
			*m_gradient
			// Also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

void Neuron::setEta(double _eta)
{
	eta = _eta;
}

void Neuron::setAlpha(double _alpha)
{
	alpha = _alpha;
}

void Neuron::setFunc(activationFunc _func)
{
	m_transferFunc = _func;
}

double Neuron::transferFunction(double x)
{
	switch (m_transferFunc)
	{
	case hypertan:
		return tanh(x);
		break;
	case sigmoid:
		return 1 / (1 + exp(-x));
		break;
	case relu:
		return log(1 + exp(x));
		break;
	}
}

double Neuron::transferFunctionDerivative(double x)
{
	switch (m_transferFunc)
	{
	case hypertan:
		return 1.0 - x * x; // (approximation)
		break;
	case sigmoid:
		return transferFunction(x)*(1 - transferFunction(x));
		break;
	case relu:
		return exp(x) / (1 + exp(x));
		break;
	}
}

double Neuron::sumDOW(const Layer & nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}
