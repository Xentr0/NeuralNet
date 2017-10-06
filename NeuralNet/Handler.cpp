#include "Handler.h"
#include "Net.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>

Handler::Handler()
{
}


Handler::~Handler()
{
}

void Handler::SetParams(Params _params)
{
	SetTrainData(_params.trainPath);
	m_params = _params;
	m_net = Net(m_params);
	m_paramsSet = 1;
}

void Handler::SetTrainData(string _path)
{
	if (!correctFormat(_path))
		exit(1); 
	if (!fileExists(_path))
		exit(1);
	m_trainpath = _path;
	m_trainSet = 1;
}

void Handler::SetTestData(string _path)
{
	if (!correctFormat(_path))
		exit(1);
	if (!fileExists(_path))
		exit(1);
	m_testpath = _path;
	m_testSet = 1;
}

void Handler::Train()
{
	if (!m_paramsSet)
	{
		exit(1);
	}
	if (!m_trainSet)
	{
		exit(1);
	}
	vector<vector<double>> inputs;
	vector<vector<double>> targets;
	getData(m_trainpath, inputs, targets);

	vector<int> indexes;
	for (unsigned i = 0; i < inputs.size(); i++)
	{
		indexes.push_back(i);
	}
	int trainingPass = 0;
	for (unsigned i = 0; i < m_params.epochs; i++)
	{
		random_shuffle(indexes.begin(), indexes.end());

		for (unsigned j = 0; j < inputs.size(); j++)
		{
			trainingPass++;

			if (trainingPass % 1000 == 0)
			{
				cout << "Pass " << trainingPass << endl;
			}

			m_net.feedForward(inputs[indexes[j]]);

			// Train the net what the outputs should have been:
			m_net.backProp(targets[indexes[j]]);
		}
	}

	m_netTrained = 1;

	if (m_writeWeights)
	{
		SetWeightsPath();
		m_net.writeWeights(m_weightspath);
	}
}

void Handler::Test()
{
	if (!m_netTrained)
	{
		exit(1);
	}
	if (!m_testSet)
	{
		exit(1);
	}
	vector<vector<double>> inputs;
	vector<vector<double>> targets;
	vector<double> results;
	getData(m_testpath, inputs, targets);
	int testPass = 0;
	int correct = 0;
	int incorrect = 0;
	for (unsigned i = 0; i < inputs.size(); i++)
	{
		++testPass;
		if (testPass % 1000 == 0)
		{
			cout << "Pass " << testPass << endl;
		}

		m_net.feedForward(inputs[i]);

		m_net.getResults(results);

		int result = 0;
		int target = 0;
		for (unsigned u = 0; u < results.size(); u++)
		{
			result = round(results[u]);
			result *= u;
		}

		vector<double> targetVals = targets[i];
		for (unsigned u = 0; u < targetVals.size(); u++)
		{
			target = u * targetVals[u];
		}

		if (result == target)
		{
			correct += 1;
		}
		else
		{
			incorrect += 1;
		}
	}

	assert(correct + incorrect == inputs.size());

	if (m_verboseResults)
	{
		cout << "Correct guesses: " << correct << endl;
		cout << "Incorrect guesses: " << incorrect << endl;
		double ratio = (double)correct / ((double)correct + (double)incorrect);
		cout << "Ratio: " << ratio << endl;
	}
	if (m_writeResults)
	{
		SetResultPath();
		ofstream outfile(m_resultspath);

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
		outfile << "-----------------------------------------------------------------------" << endl;
		outfile << "Correct guesses: " << correct << endl;
		outfile << "Incorrect guesses: " << incorrect << endl;
		double ratio = (double)correct / ((double)correct + (double)incorrect);
		outfile << "Ratio: " << ratio << endl;
	}
}

void Handler::SetWeights(string _path)
{
	// Check if weights file exists
	if (!fileExists(_path))
		exit(1);
	// Check if the weights file has the correct extension
	if (_path.find(".weights") == string::npos)
		exit(1);
	// Check format and import parameters
	ifstream file;
	file.open(_path.c_str());

	string line;
	getline(file, line);
	stringstream ss(line);
	string label;

	ss >> label;
	if (label.compare("traindata:") != 0)
		exit(1);
	string trainPath;
	ss >> trainPath;
	m_params.trainPath = trainPath;

	getline(file, line);
	ss.swap(stringstream(line));
	ss >> label;
	if (label.compare("topology:") != 0)
		exit(1);
	double oneValue;
	vector<unsigned> topology;
	while (ss >> oneValue)
	{
		topology.push_back(oneValue);
	}
	m_params.topology = topology;

	getline(file, line);
	ss.swap(stringstream(line));
	ss >> label;
	if (label.compare("eta:") != 0)
		exit(1);
	double eta;
	ss >> eta;
	m_params.eta = eta;

	getline(file, line);
	ss.swap(stringstream(line));
	ss >> label;
	if (label.compare("alpha:") != 0)
		exit(1);
	double alpha;
	ss >> alpha;
	m_params.alpha = alpha;

	getline(file, line);
	ss.swap(stringstream(line));
	ss >> label;
	if (label.compare("activationfunction:") != 0)
		exit(1);
	string func;
	ss >> func;
	activationFunc transferFunc;
	if (func == "hypertan")
		transferFunc = hypertan;
	if (func == "sigmoid")
		transferFunc = sigmoid;
	if (func == "relu")
		transferFunc = relu;
	m_params.func = transferFunc;

	getline(file, line);
	ss.swap(stringstream(line));
	ss >> label;
	if (label.compare("epochs:") != 0)
		exit(1);
	int epochs;
	ss >> epochs;
	m_params.epochs = epochs;

	getline(file, line);
	ss.swap(stringstream(line));
	ss >> label;
	if (label.compare("batchsize:") != 0)
		exit(1);
	int batchSize;
	ss >> batchSize;
	m_params.batchSize = batchSize;

	// Put the actual weights in a vector
	vector<double> weights;
	while (true)
	{
		getline(file, line);
		ss.swap(stringstream(line));

		double weight;
		ss >> weight;
		if (file.eof())
			break;
		weights.push_back(weight);
	}

	// Make sure weights vector has the right size compared to our desired topology
	int desiredSize = 0;
	for (unsigned i = 0; i < topology.size()-1; i++)
	{
		desiredSize += (topology[i] +1) * topology[i+1]; //+1 adds the bias nodes for which we have weights but that are not added in the topology
	}	
	assert(weights.size() == desiredSize);

	// Make our Net object and set the desired weights
	m_net = Net(m_params);
	m_paramsSet = 1;

	m_net.copyWeights(weights);
	m_netTrained = 1;	
}

void Handler::getData(string _path, vector<vector<double>>& _inputs, vector<vector<double>>& _targets)
{
	ifstream file;
	file.open(_path.c_str());
	string line;

	int iter = 0;

	while (!file.eof())
	{
		vector<double> inputVals;
		vector<double> targetVals;

		getline(file, line);
		stringstream ss(line);

		string label;
		ss >> label;
		if (label.compare("in:") == 0)
		{
			double oneValue;
			while (ss >> oneValue)
			{
				inputVals.push_back(oneValue);
			}
			_inputs.push_back(inputVals);
		}

		getline(file, line);
		stringstream ss1(line);

		ss1 >> label;
		if (label.compare("out:") == 0)
		{
			double oneValue;
			while (ss1 >> oneValue)
			{
				targetVals.push_back(oneValue);
			}
			_targets.push_back(targetVals);
		}
		iter++;
	}

}

void Handler::SetResultPath()
{
	string testpath = m_testpath;
	m_resultspath = testpath.replace(testpath.end() - 4, testpath.end(), ".results0");
	int i = 1;
	while (fileExists(m_resultspath))
	{
		string itr = to_string(i);
		m_resultspath = m_resultspath.replace(m_resultspath.end() - 1, m_resultspath.end(), itr);
		i++;
	}
}

void Handler::SetWeightsPath()
{
	string trainpath = m_trainpath;
	m_weightspath = trainpath.replace(trainpath.end() - 4, trainpath.end(), ".weights0");
	int i = 1;
	while (fileExists(m_weightspath))
	{
		string itr = to_string(i);
		m_weightspath = m_weightspath.replace(m_weightspath.end() - 1, m_weightspath.end(), itr);
		i++;
	}
}

bool Handler::correctFormat(string path)
{
	string format = ".txt";
	if (path.length() >= format.length())
	{
		return (0 == path.compare(path.length() - format.length(), format.length(), format));
	}
	else
	{
		return false;
	}
}

bool Handler::fileExists(string& path)
{
	if (FILE *file = fopen(path.c_str(), "r"))
	{
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}
