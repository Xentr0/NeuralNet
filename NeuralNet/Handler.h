#pragma once
#include "Net.h"
#include "MyVars.h"
#include <vector>
#include <string>

using namespace std;

class Handler
{
public:
	Handler();
	~Handler();
	
	void SetParams(Params _params);	
	void SetTestData(string _path);

	void Train();
	void Test();

	void SetWeights(string _path);

	void WriteWeights(bool _write) { m_writeWeights = _write; };
	void WriteResults(bool _write) { m_writeResults = _write; };
	void VerboseResults(bool _verbose) { m_verboseResults = _verbose; };

private:
	void SetTrainData(string _path);
	void getData(string _path, vector<vector<double>>& _inputs, vector<vector<double>>& _targets);
	void SetResultPath();
	void SetWeightsPath();

	string m_trainpath;
	string m_weightspath;
	string m_resultspath;
	string m_testpath;

	Params m_params;
	Net m_net = Net(m_params);

	bool m_paramsSet = 0;
	bool m_netTrained = 0;
	bool m_trainSet = 0;
	bool m_testSet = 0;

	bool m_writeWeights = 1;
	bool m_writeResults = 1;
	bool m_verboseResults = 1;

	bool correctFormat(string path);
	bool fileExists(string& path);
};

