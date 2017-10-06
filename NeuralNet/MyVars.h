#pragma once
#include <vector>
using namespace std;

enum activationFunc { sigmoid, hypertan, relu };
struct Params
{
	vector<unsigned> topology;
	double eta;
	double alpha;
	activationFunc func;
	int epochs;
	int batchSize;
	string trainPath;
};
struct Connection
{
	double weight;
	double deltaWeight;
};