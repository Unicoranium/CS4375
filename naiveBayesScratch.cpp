#define _USE_MATH_DEFINES

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

double calc_age_lh(double age, double mean, double var)
{
	return 1 / sqrt(2 * M_PI * var) * exp(-pow(age - mean, 2) / (2 * var));
}
VectorXd calc_raw_prob(int pclass, int sex, int age, MatrixXd lh_pclass, MatrixXd lh_sex, VectorXd apriori, VectorXd age_mean, VectorXd age_var)
{
	double num_s = lh_pclass(1, pclass - 1) * lh_sex(1, sex) * apriori[1] * calc_age_lh(age, age_mean[1], age_var[1]);
	double num_p = lh_pclass(0, pclass - 1) * lh_sex(0, sex) * apriori[0] * calc_age_lh(age, age_mean[0], age_var[0]);
	double den = num_s + num_p;
	VectorXd res(2);
	res << num_p / den, num_s / den;
	return res;
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

	VectorXd apriori = VectorXd(2);

	MatrixXd data_matrix(800, 2);
	VectorXd labels = VectorXd(800);
	int s = 0;
	for (int i = 0; i < 800; i++)
	{
		data_matrix.row(i) << 1, sex[i];
		labels.row(i) << survived[i];
		s += survived[i];
	}

	apriori << (double)(numObservations - survived.sum()) / numObservations, (double)(survived.sum()) / numObservations;
	cout << "Apriori: " << apriori << endl;

	VectorXd count_survived = VectorXd(2);
	count_survived << numObservations - survived.sum(), survived.sum();

	MatrixXd lh_pclass(2, 3);
	lh_pclass << 0, 0, 0, 0, 0, 0;

	for (int sv = 0; sv < 2; sv++)
	{
		for (int pc = 0; pc < 3; pc++)
		{
			int total = 0;
			for (int i = 0; i < numObservations; i++)
			{
				if (survived[i] == sv && pclass[i] == pc + 1)
				{
					total++;
				}
			}
			lh_pclass(sv, pc) = total / count_survived(sv);
		}
	}
	MatrixXd lh_sex(2, 2);
	lh_sex << 0, 0, 0, 0;

	for (int sv = 0; sv < 2; sv++)
	{
		for (int sx = 0; sx < 2; sx++)
		{
			int total = 0;
			for (int i = 0; i < numObservations; i++)
			{
				if (survived[i] == sv && sex[i] == sx)
				{
					total++;
				}
			}
			lh_sex(sv, sx) = total / count_survived(sv);
		}
	}

	cout << "Likelihood values for p(pclass|survived):" << lh_pclass << endl;
	cout << "Likelihood values for p(sex|survived):" << lh_sex << endl;

	VectorXd age_mean(2);
	VectorXd age_var(2);

	age_mean << 0, 0;
	age_var << 0, 0;

	for (int i = 0; i < numObservations; i++)
	{
		age_mean[survived[i]] += age[i];
	}
	age_mean[0] /= count_survived[0];
	age_mean[1] /= count_survived[1];
	for (int i = 0; i < numObservations; i++)
	{
		age_var[survived[i]] += pow(age[i] - age_mean[survived[i]], 2);
	}
	age_var[0] /= count_survived[0] - 1;
	age_var[1] /= count_survived[1] - 1;

	cout << "Mean age for survived" << age_mean << endl;
	cout << "Variance age for survived" << age_var << endl;

	// ----

	MatrixXd test_matrix = MatrixXd(numObservations - 800, 2);
	VectorXd test_labels = VectorXd(numObservations - 800);

	for (int i = 800; i < numObservations; i++)
	{
		test_matrix.row(i - 800) << 1, sex[i];
		test_labels.row(i - 800) << survived[i];
	}

	int correct = 0;
	int total = 0;

	int TP = 0;
	int TN = 0;
	int FP = 0;
	int FN = 0;

	for (int i = 800; i < numObservations; i++)
	{
		VectorXd raw = calc_raw_prob(pclass[i], sex[i], age[i], lh_pclass, lh_sex, apriori, age_mean, age_var);
		int choose = raw[1] > 0.5 ? 1 : 0;
		if (choose == survived[i])
		{
			correct++;
		}
		total++;
		if (choose == 1 && survived[i] == 1)
		{
			TP++;
		}
		else if (choose == 0 && survived[i] == 0)
		{
			TN++;
		}
		else if (choose == 1 && survived[i] == 0)
		{
			FP++;
		}
		else if (choose == 0 && survived[i] == 1)
		{
			FN++;
		}
	}

	cout << "Accuracy: " << (double)correct / total << endl;
	cout << "Sensitivity: " << (double)TP / (TP + FN) << endl;
	cout << "Specificity: " << (double)TN / (TN + FP) << endl;
	auto t2 = high_resolution_clock::now();

	/* Getting number of milliseconds as an integer. */
	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;

	std::cout << ms_int.count() << "ms\n";
	std::cout << ms_double.count() << "ms\n";
	cout << "\nProgram Terminated." << endl;

	return 0;
}