#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <ColumnWiseMatrix.h>
#include <Vector.h>

#include <AdvectionDiffusionSolver1D.h>
#include <AdvectionDiffusionSolver2D.h>
#include <WaveEquationSolver1D.h>
#include <WaveEquationSolver2D.h>

#include <Exception.h>
#include <Utils/CommandLineParser.h>
#include <Utils/EnumParser.h>

#define DEBUG_PRINT_START(X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \
	if (debug)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
	{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
		start = std::chrono::high_resolution_clock::now();                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     \
		std::cout << #X << std::endl;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
	}

#define DEBUG_PRINT_END                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
	if (debug)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
	{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
		end = std::chrono::high_resolution_clock::now();                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
		auto elapsedTime = 1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              \
		std::cout << "... Done"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
				  << "[ " << elapsedTime << " ms ]" << std::endl;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              \
	}

#define DEBUG_PRINT(X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
	if (debug)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
		std::cout << #X << std::endl;

template<class solverImpl, MathDomain md>
std::unique_ptr<solverImpl> setup1D(const clp::CommandLineArgumentParser& ap, const bool debug)
{
	using vType = cl::Vector<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	std::chrono::high_resolution_clock::time_point start, end;

	DEBUG_PRINT_START(Parsing inputs...)

#pragma region Parse Inputs

	auto initialConditionFileString = ap.GetArgumentValue<std::string>("-ic", "");
	auto gridFileString = ap.GetArgumentValue<std::string>("-g", "");

	auto leftBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-lbct");
	auto leftBoundaryConditionType = ep::ParseBoundaryConditionType(leftBoundaryConditionTypeString);
	auto leftBoundaryConditionValue = ap.GetArgumentValue<sType>("-lbc");

	std::string rightBoundaryConditionTypeString = ap.GetArgumentValue("-rbct", leftBoundaryConditionTypeString);
	auto rightBoundaryConditionType = ep::ParseBoundaryConditionType(rightBoundaryConditionTypeString);
	auto rightBoundaryConditionValue = ap.GetArgumentValue("-rbc", leftBoundaryConditionValue);

	auto solverType = ep::ParseSolverType(ap.GetArgumentValue<std::string>("-st", "CrankNicolson"));
	auto spaceDiscretizerType = ep::ParseSpaceDiscretizer(ap.GetArgumentValue<std::string>("-sdt", "Upwind"));
	if (spaceDiscretizerType == SpaceDiscretizerType::LaxWendroff && solverType != SolverType::ExplicitEuler)
	{
		std::cout << "WARNING: Lax-Wendroff scheme can be applied only with ExplicitEuler -> overriding solver type" << std::endl;
		solverType = SolverType::ExplicitEuler;
	}

	auto diffusion = ap.GetArgumentValue<sType>("-d");
	auto velocity = ap.GetArgumentValue<sType>("-v");
	auto dt = ap.GetArgumentValue<sType>("-dt");

#pragma endregion

	DEBUG_PRINT_END

	using vType = cl::Vector<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	DEBUG_PRINT_START(Creating grid...)

	std::ifstream gridFile(gridFileString);
	std::unique_ptr<vType> grid = nullptr;
	if (!gridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128))
		grid = std::make_unique<vType>(vType::LinSpace(sType(-4.0), sType(4.0), 128u));
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		grid = std::make_unique<vType>(vType::VectorFromBinaryFile(gridFileString));
	}
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating initial condition...)

	std::ifstream initialConditionFile(initialConditionFileString);
	std::unique_ptr<vType> initialCondition = nullptr;
	if (!initialConditionFile.is_open())
	{
		DEBUG_PRINT(... creating bell function)
		auto _grid = grid->Get();
		std::vector<sType> bellFunction(grid->size());
		for (unsigned i = 0; i < bellFunction.size(); ++i)
			bellFunction[i] = std::exp(static_cast<sType>(-.25) * _grid[i] * _grid[i]);

		initialCondition = std::make_unique<vType>(bellFunction);
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		initialCondition = std::make_unique<vType>(vType::VectorFromBinaryFile(initialConditionFileString));
	}
	DEBUG_PRINT_END

	BoundaryCondition leftBc(leftBoundaryConditionType, static_cast<double>(leftBoundaryConditionValue));
	BoundaryCondition rightBc(rightBoundaryConditionType, static_cast<double>(rightBoundaryConditionValue));
	BoundaryCondition1D bc(leftBc, rightBc);

	DEBUG_PRINT_START(Creating PDE input data...)
	pde::PdeInputData1D<MemorySpace::Device, md> data(*initialCondition, *grid, velocity, diffusion, dt, solverType, spaceDiscretizerType, bc);
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating PDE solver...)
	auto solver = std::make_unique<solverImpl>(std::move(data));
	DEBUG_PRINT_END

	return solver;
}

template<class solverImpl, MathDomain md>
std::unique_ptr<solverImpl> setup2D(const clp::CommandLineArgumentParser& ap, const bool debug)
{
	using vType = cl::Vector<MemorySpace::Device, md>;
	using mType = cl::ColumnWiseMatrix<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

	DEBUG_PRINT_START(Parsing inputs...)

#pragma region Parse Inputs

	auto initialConditionFileString = ap.GetArgumentValue<std::string>("-ic", "");
	auto xGridFileString = ap.GetArgumentValue<std::string>("-gx", "");
	auto yGridFileString = ap.GetArgumentValue<std::string>("-gy", "");

#pragma region BC

	auto leftBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-lbct");
	auto leftBoundaryConditionType = ep::ParseBoundaryConditionType(leftBoundaryConditionTypeString);
	sType leftBoundaryConditionValue = ap.GetArgumentValue<sType>("-lbc");

	std::string rightBoundaryConditionTypeString = ap.GetArgumentValue("-rbct", leftBoundaryConditionTypeString);
	auto rightBoundaryConditionType = ep::ParseBoundaryConditionType(rightBoundaryConditionTypeString);
	sType rightBoundaryConditionValue = ap.GetArgumentValue("-rbc", leftBoundaryConditionValue);

	auto downBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-dbct", leftBoundaryConditionTypeString);
	auto downBoundaryConditionType = ep::ParseBoundaryConditionType(downBoundaryConditionTypeString);
	sType downBoundaryConditionValue = ap.GetArgumentValue<sType>("-dbc", leftBoundaryConditionValue);

	auto upBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-ubct", leftBoundaryConditionTypeString);
	auto upBoundaryConditionType = ep::ParseBoundaryConditionType(upBoundaryConditionTypeString);
	sType upBoundaryConditionValue = ap.GetArgumentValue<sType>("-ubc", leftBoundaryConditionValue);

#pragma endregion

	auto solverType = ep::ParseSolverType(ap.GetArgumentValue<std::string>("-st", "CrankNicolson"));
	auto spaceDiscretizerType = ep::ParseSpaceDiscretizer(ap.GetArgumentValue<std::string>("-sdt", "Upwind"));
	if (spaceDiscretizerType == SpaceDiscretizerType::LaxWendroff && solverType != SolverType::ExplicitEuler)
	{
		std::cout << "WARNING: Lax-Wendroff scheme can be applied only with ExplicitEuler -> overriding solver type" << std::endl;
		solverType = SolverType::ExplicitEuler;
	}

	auto diffusion = ap.GetArgumentValue<sType>("-d");
	auto xVelocity = ap.GetArgumentValue<sType>("-vx");
	auto yVelocity = ap.GetArgumentValue<sType>("-vy");
	auto dt = ap.GetArgumentValue<sType>("-dt");

#pragma endregion

	DEBUG_PRINT_END

#pragma region Grid

	DEBUG_PRINT_START(Creating x grid...)

	std::ifstream xGridFile(xGridFileString);
	std::unique_ptr<vType> xGrid = nullptr;
	constexpr unsigned defaultSize = 10u;
	if (!xGridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128))
		xGrid = std::make_unique<vType>(vType::LinSpace(sType(-4.0), sType(4.0), defaultSize));
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		xGrid = std::make_unique<vType>(vType::VectorFromBinaryFile(xGridFileString));
	}
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating y grid...)

	std::ifstream yGridFile(yGridFileString);
	std::unique_ptr<vType> yGrid = nullptr;
	if (!yGridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128))
		yGrid = std::make_unique<vType>(vType::LinSpace(sType(-4.0), sType(4.0), defaultSize));
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		yGrid = std::make_unique<vType>(vType::VectorFromBinaryFile(yGridFileString));
	}
	DEBUG_PRINT_END

#pragma endregion

	DEBUG_PRINT_START(Creating initial condition...)

	std::ifstream initialConditionFile(initialConditionFileString);
	std::unique_ptr<mType> initialCondition = nullptr;
	if (!initialConditionFile.is_open())
	{
		DEBUG_PRINT(... creating bell function)
		auto _xGrid = xGrid->Get();
		auto _yGrid = yGrid->Get();
		std::vector<sType> bellFunction(xGrid->size() * yGrid->size());
		for (unsigned j = 0; j < _yGrid.size(); ++j)
			for (unsigned i = 0; i < _xGrid.size(); ++i)
				bellFunction[i + _xGrid.size() * j] = std::exp(static_cast<sType>(-.25) * (_xGrid[i] * _xGrid[i] + _yGrid[j] * _yGrid[j]));

		initialCondition = std::make_unique<mType>(bellFunction, xGrid->size(), yGrid->size());
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		initialCondition = std::make_unique<mType>(mType::MatrixFromBinaryFile(initialConditionFileString));
	}
	DEBUG_PRINT_END

	BoundaryCondition leftBc(leftBoundaryConditionType, static_cast<double>(leftBoundaryConditionValue));
	BoundaryCondition rightBc(rightBoundaryConditionType, static_cast<double>(rightBoundaryConditionValue));
	BoundaryCondition downBc(downBoundaryConditionType, static_cast<double>(downBoundaryConditionValue));
	BoundaryCondition upBc(upBoundaryConditionType, static_cast<double>(upBoundaryConditionValue));
	BoundaryCondition2D bc(leftBc, rightBc, downBc, upBc);

	DEBUG_PRINT_START(Creating PDE input data...)
	pde::PdeInputData2D<MemorySpace::Device, md> data(*initialCondition, *xGrid, *yGrid, xVelocity, yVelocity, diffusion, dt, solverType, spaceDiscretizerType, bc);
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating PDE solver...)
	auto solver = std::make_unique<solverImpl>(std::move(data));
	DEBUG_PRINT_END

	return solver;
}