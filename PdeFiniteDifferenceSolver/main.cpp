// Main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <fstream>
#include <algorithm>

#include <Exception.h>

#pragma region Command Line Parser

class IllegalArgumentException : public Exception
{
public:
	IllegalArgumentException(const std::string& message = "")
		: Exception("IllegalArgumentException: " + message)
	{
	}
};

class ArgumentNotFoundException : public Exception
{
public:
	ArgumentNotFoundException(const std::string& message = "")
		: Exception("ArgumentNotFoundException: " + message)
	{
	}
};

class CommandLineArgumentParser
{
public:
	CommandLineArgumentParser(int argc, char **argv)
		: args(argv, argv + argc)
	{
	}

	template<typename T>
	T GetArgumentValue(const std::string& option) const;

	template<typename T>
	T GetArgumentValue(const std::string& option, const T& default) const noexcept
	{
		T ret;
		try
		{
			ret = GetArgumentValue<T>(option);
		}
		catch (ArgumentNotFoundException&)
		{
			ret = default;
		}

		return ret;
	}

	bool GetFlag(const std::string& option) const
	{
		return std::find(args.begin(), args.end(), option) != args.end();
	}

private:
	std::vector<std::string> args;
};

template<>
std::string CommandLineArgumentParser::GetArgumentValue<std::string>(const std::string& option) const
{
	auto itr = std::find(args.begin(), args.end(), option);
	if (itr != args.end())
	{
		if (++itr == args.end())
			throw IllegalArgumentException(option);
		return *itr;
	}

	throw ArgumentNotFoundException(option);
}

template<>
int CommandLineArgumentParser::GetArgumentValue<int>(const std::string& option) const
{
	return std::atoi(GetArgumentValue<std::string>(option).c_str());
}

template<>
double CommandLineArgumentParser::GetArgumentValue<double>(const std::string& option) const
{
	return std::atof(GetArgumentValue<std::string>(option).c_str());
}

#pragma endregion

#pragma region Enum Mapping

#define PARSE(E, X)\
	if (!strcmp(text.c_str(), #X))\
		return E::X;

SolverType parseSolverType(const std::string& text)
{
#define PARSE_WORKER(X) PARSE(SolverType, X);

	PARSE_WORKER(ExplicitEuler);
	PARSE_WORKER(ImplicitEuler);
	PARSE_WORKER(CrankNicolson);
	PARSE_WORKER(RungeKuttaRalston);
	PARSE_WORKER(RungeKutta3);
	PARSE_WORKER(RungeKutta4);
	PARSE_WORKER(RungeKuttaThreeEight);
	PARSE_WORKER(RungeKuttaGaussLegendre4);
	PARSE_WORKER(RichardsonExtrapolation2);
	PARSE_WORKER(RichardsonExtrapolation3);
	PARSE_WORKER(AdamsBashforth2);
	PARSE_WORKER(AdamsMouldon2);

#undef PARSE_WORKER

	return SolverType::Null;
}

BoundaryConditionType parseBoundaryConditionType(const std::string& text)
{
#define PARSE_WORKER(X) PARSE(BoundaryConditionType, X);

	PARSE_WORKER(Dirichlet);
	PARSE_WORKER(Neumann);
	PARSE_WORKER(Periodic);

#undef PARSE_WORKER
	return BoundaryConditionType::Null;
}

#undef PARSE_WORKER

#pragma endregion

int main(int argc, char** argv)
{
#pragma region Parse Inputs

	CommandLineArgumentParser ap(argc, argv);

	auto initialConditionFileString = ap.GetArgumentValue<std::string>("-ic", "");
	auto gridFileString = ap.GetArgumentValue<std::string>("-g", "");
	auto outputFileString = ap.GetArgumentValue<std::string>("-of", "sol.cl");

	auto leftBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-lbct");
	auto leftBoundaryConditionType = parseBoundaryConditionType(leftBoundaryConditionTypeString);
	double leftBoundaryConditionValue = ap.GetArgumentValue<double>("-lbc");

	std::string rightBoundaryConditionTypeString = ap.GetArgumentValue("rbct", leftBoundaryConditionTypeString);
	auto rightBoundaryConditionType = parseBoundaryConditionType(rightBoundaryConditionTypeString);
	double rightBoundaryConditionValue = ap.GetArgumentValue("-rbc", leftBoundaryConditionValue);

	auto solverType = parseSolverType(ap.GetArgumentValue<std::string>("-st", "CrankNicolson"));

	auto diffusion = ap.GetArgumentValue<double>("-d");
	auto velocity = ap.GetArgumentValue<double>("-v");
	auto dt = ap.GetArgumentValue<double>("-dt");

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<int>("-n");

	// total number of steps
	auto N = ap.GetArgumentValue<int>("-N");

#pragma endregion

	std::ifstream gridFile(gridFileString);
	cl::vec *grid = nullptr;
	if (!gridFile.is_open())
		grid = new cl::vec(cl::LinSpace(0.0f, 1.0f, 128));
	else
		grid = new cl::vec(cl::DeserializeVector(gridFile));

	std::ifstream initialConditionFile(initialConditionFileString);
	cl::vec *initialCondition = nullptr;
	if (!initialConditionFile.is_open())
	{
		auto _grid = grid->Get();
		std::vector<float> bellFunction(grid->size());
		for (unsigned i = 0; i < bellFunction.size(); ++i)
			bellFunction[i] = exp(-.25 * _grid[i] * _grid[i]);

		initialCondition = new cl::vec(bellFunction);
	}
	else
		initialCondition = new cl::vec(cl::DeserializeVector(initialConditionFile));

	pde::GpuSinglePdeInputData data(*initialCondition, *grid, velocity, diffusion, dt, solverType);
	pde::sol1D solver(data);

	std::vector<float> solutionMatrix;

	unsigned nSolutions = 0;
	for (unsigned m = 0; m < N; ++m)
	{
		solver.Advance(n);
		const auto solution = solver.solution->columns[0]->Get();

		solutionMatrix.insert(solutionMatrix.end(), solution.begin(), solution.end());
		++nSolutions;
	}

	std::ofstream outputFile(outputFileString);
	cl::SerializeMatrix(solutionMatrix, initialCondition->size(), nSolutions, outputFile);

	delete initialCondition;
	delete grid;
}

