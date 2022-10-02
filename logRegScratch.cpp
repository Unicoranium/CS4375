#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <Eigen/Dense>
#include <chrono>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

VectorXd sigmoid(VectorXd x)
{
	VectorXd result(x.size());
	for (int i = 0; i < x.size(); i++)
	{
		result[i] = 1 / (1 + exp(-x(i)));
	}
	return result;
}

VectorXd prob(VectorXd x)
{
	VectorXd result(x.size());
	for (int i = 0; i < x.size(); i++)
	{
		result[i] = exp(x(i)) / (1 + exp(x(i)));
	}
	return result;
}

int main(int argc, char **arv)
{
	auto t1 = high_resolution_clock::now();
	ifstream inFS;
	string line;
	string dump;
	string pclass_in, survived_in, sex_in, age_in;
	const int MAX_LEN = 1100;
	VectorXd pclass = VectorXd(MAX_LEN);
	VectorXd survived = VectorXd(MAX_LEN);
	VectorXd sex = VectorXd(MAX_LEN);
	VectorXd age = VectorXd(MAX_LEN);

	cout << "Opening file titanic_project.csv" << endl;

	inFS.open("titanic_project.csv");
	if (!inFS.is_open())
	{
		cout << "Could not open file titanic_project.csv" << endl;
		return 1;
	}

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	cout << "Heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good())
	{

		getline(inFS, dump, ',');
		getline(inFS, pclass_in, ',');
		getline(inFS, survived_in, ',');
		getline(inFS, sex_in, ',');
		getline(inFS, age_in, '\n');

		pclass[numObservations] = stoi(pclass_in);
		survived[numObservations] = stoi(survived_in);
		sex[numObservations] = stoi(sex_in);
		age[numObservations] = stoi(age_in);

		// data.at(numObservations) = {stoi(pclass_in), stoi(survived_in), stoi(sex_in), stoi(age_in)};

		numObservations++;
	}

	pclass.conservativeResize(numObservations);
	survived.conservativeResize(numObservations);
	sex.conservativeResize(numObservations);
	age.conservativeResize(numObservations);

	cout << "Closing file titanic_project.csv" << endl;
	inFS.close();

	VectorXd weights = VectorXd(2);
	weights << 1, 1;

	MatrixXd data_matrix(800, 2);
	VectorXd labels = VectorXd(800);

	for (int i = 0; i < 800; i++)
	{
		data_matrix.row(i) << 1, sex[i];
		labels.row(i) << survived[i];
	}

	VectorXd prob_vector = VectorXd(800);
	VectorXd error = VectorXd(800);

	for (int i = 0; i < 1000; i++)
	{
		prob_vector = sigmoid(data_matrix * weights);
		error = labels - prob_vector;
		weights = weights + data_matrix.transpose() * error;
	}

	MatrixXd test_matrix = MatrixXd(numObservations - 800, 2);
	VectorXd test_labels = VectorXd(numObservations - 800);

	for (int i = 800; i < numObservations; i++)
	{
		test_matrix.row(i - 800) << 1, sex[i];
		test_labels.row(i - 800) << survived[i];
	}

	VectorXd predicted = test_matrix * weights;
	VectorXd probabilities = prob(predicted);

	int correct = 0;
	int total = 0;

	int TP = 0;
	int TN = 0;
	int FP = 0;
	int FN = 0;

	for (int i = 0; i < numObservations - 800; i++)
	{
		if (round(probabilities[i]) == test_labels[i])
		{
			correct++;
		}
		total++;
		if (round(probabilities[i]) == 1 && test_labels[i] == 1)
		{
			TP++;
		}
		else if (round(probabilities[i]) == 0 && test_labels[i] == 0)
		{
			TN++;
		}
		else if (round(probabilities[i]) == 1 && test_labels[i] == 0)
		{
			FP++;
		}
		else if (round(probabilities[i]) == 0 && test_labels[i] == 1)
		{
			FN++;
		}
	}

	cout << "Weights: " << weights << endl;

	cout << "Accuracy: " << (double)correct / total << endl;
	cout << "Sensitivity: " << (double)TP / (TP + FN) << endl;
	cout << "Specificity: " << (double)TN / (TN + FP) << endl;

	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;

	cout << ms_int.count() << "ms\n";
	cout << ms_double.count() << "ms\n";

	cout << "\nProgram Terminated." << endl;

	return 0;
}