#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <array>

using namespace std;

double sum(vector<double> &v)
{
	double sum = 0;
	for (int i = 0; i < (int)v.size(); i++)
	{
		sum += v[i];
	}
	return sum;
}

double mean(vector<double> &v)
{
	return sum(v) / v.size();
}

double median(vector<double> &v)
{
	if (v.size() % 2 == 0)
	{
		return (v[v.size() / 2] + v[v.size() / 2 - 1]) / 2;
	}
	else
	{
		return v[v.size() / 2];
	}
}

array<double, 2> range(vector<double> &v)
{
	array<double, 2> res = {v[0], v[0]};
	for (int i = 0; i < (int)v.size(); i++)
	{
		if (v[i] < res[0])
		{
			res[0] = v[i];
		}
		if (v[i] > res[1])
		{
			res[1] = v[i];
		}
	}
	return res;
}

void print_stats(vector<double> &data)
{
	cout << "Sum: " << sum(data) << endl;
	cout << "Mean: " << mean(data) << endl;
	cout << "Median: " << median(data) << endl;
	array<double, 2> r = range(data);
	cout << "Range: " << r[0] << " - " << r[1] << endl;
}

float covar(vector<double> &x, vector<double> &y)
{
	double x_mean = mean(x);
	double y_mean = mean(y);
	double sum = 0;
	for (int i = 0; i < (int)x.size(); i++)
	{
		sum += (x[i] - x_mean) * (y[i] - y_mean);
	}
	return sum / (x.size() - 1);
}

float cor(vector<double> &x, vector<double> &y)
{
	return covar(x, y) / (sqrt(covar(x, x)) * sqrt(covar(y, y)));
}

int main(int argc, char **arv)
{
	ifstream inFS;
	string line;
	string rm_in, medv_in;
	const int MAX_LEN = 1000;
	vector<double> rm(MAX_LEN);
	vector<double> medv(MAX_LEN);

	cout << "Opening file Boston.csv" << endl;

	inFS.open("Boston.csv");
	if (!inFS.is_open())
	{
		cout << "Could not open file Boston.csv" << endl;
		return 1;
	}

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	cout << "Heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good())
	{

		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');

		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);

		numObservations++;
	}

	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "New Length: " << rm.size() << endl;

	cout << "Closing file Boston.csv" << endl;
	inFS.close();

	cout << "Number of records: " << numObservations << endl;

	cout << "\nStats for rm" << endl;
	print_stats(rm);

	cout << "\nStats for medv" << endl;
	print_stats(medv);

	cout << "\nCovariance = " << covar(rm, medv) << endl;

	cout << "\nCorrelation = " << cor(rm, medv) << endl;

	cout << "\nProgram Terminated." << endl;

	return 0;
}