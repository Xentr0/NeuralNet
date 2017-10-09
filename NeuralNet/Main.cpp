#include "Handler.h"
#include "MyVars.h"
#include <string>
#include <vector>

int main()
{
	string trainPath = "/home/xentro/Projects/ConvertMnist/ConvertMnist/Data/trainMNIST.txt";
	string testPath  = "/home/xentro/Projects/ConvertMnist/ConvertMnist/Data/testMNIST.txt";
	string weightsPath = "/home/xentro/Projects/ConvertMnist/ConvertMnist/Data/trainMNIST.weights0";

	vector<unsigned> topology;
	topology.push_back(784);
	topology.push_back(100);
	topology.push_back(10);

	Params myParams;
	myParams.topology = topology;

	// Hard coded params to be made into user input
	myParams.alpha = 0;
	myParams.eta = 3;
	myParams.func = activationFunc::sigmoid;
	myParams.epochs = 3;
	myParams.batchSize = 10;
	myParams.trainPath = trainPath;

	Handler myHandler;
	// Setting required data for handler
	myHandler.SetParams(myParams);
	myHandler.SetTestData(testPath);

	// Specifying which outputs we want
	myHandler.WriteWeights(1);
	myHandler.WriteResults(1);
	myHandler.VerboseResults(1);

	// Actually training and testing the network
	myHandler.Train();
	myHandler.Test();
	myHandler.SetWeights(weightsPath);
	myHandler.Test();
	
	return 0;
}
