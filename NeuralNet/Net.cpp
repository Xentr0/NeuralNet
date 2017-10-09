#include "Net.h"
#include <cassert>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

Net::Net(const Params _params)
{
	m_params = _params;

	// Make the actual neural network
	unsigned numLayers = m_params.topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == m_params.topology.size() - 1 ? 0 : m_params.topology[layerNum + 1]; // If layernum is the last layer it is zero. Otherwise it is the number of elements in the next layer.

		// We have made a new Layer, now fill it with neurons,
		// and add a bias neuron to the layer:
		for (unsigned neuronNum = 0; neuronNum <= m_params.topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			m_layers.back().back().setAlpha(m_params.alpha);
			m_layers.back().back().setEta(m_params.eta);
			m_layers.back().back().setFunc(m_params.func);
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

Net::~Net()
{
}

void Net::feedForward(const vector<double>& inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double>& targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta / 2; // division by 2 is optional
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Calculate output layer gradients

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update all connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(vector<double>& resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::writeWeights(string path)
{
	ofstream outfile(path);

	outfile << "traindata: " << m_params.trainPath << endl;

	outfile << "topology: ";
	for (unsigned i = 0; i < m_params.topology.size(); ++i)
	{
		outfile << m_params.topology[i] << " ";
	}
	outfile << endl;
	outfile << "eta: " << m_params.eta << endl;
	outfile << "alpha: " << m_params.alpha << endl;

	string functionname = "";
	switch (m_params.func)
	{
	case activationFunc::hypertan:
		functionname = "hypertan";
		break;
	case activationFunc::sigmoid:
		functionname = "sigmoid";
		break;
	case activationFunc::relu:
		functionname = "relu";
		break;
	}

	outfile << "activationfunction: " << functionname << endl;

	outfile << "epochs: " << m_params.epochs << endl;
	outfile << "batchsize: " << m_params.batchSize << endl;

	for (unsigned m = 0; m < m_layers.size() - 1; m++) // -1 because output layers has no connections
	{
		for (unsigned n = 0; n < m_layers[m].size(); n++) // no -1 since we want the weights from bias nodes as well
		{
			for (unsigned i = 0; i < m_layers[m + 1].size() - 1; i++) // -1 because the previous layer is not connected to the next bias node (last node in the next layer)
			{
				outfile << m_layers[m][n].getOutputWeights()[i].weight << endl;
				int nextlayer = m + 1;
			}
		}
	}
}

void Net::copyWeights(const vector<double> &weights)
{
	unsigned k = 0;
	for (unsigned i = 0; i < m_layers.size() - 1; i++)
	{
		for (unsigned j = 0; j < m_layers[i].size(); j++)
		{
			vector<Connection> connections;
			for (unsigned u = 0; u < m_layers[i+1].size()-1; u++)
			{
				Connection connection;
				connection.weight = weights[k];
				connections.push_back(connection);
				k++;
			}
			m_layers[i][j].setOutputWeights(connections);
		}
	}
}


