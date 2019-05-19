// Main.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <algorithm>
#include <chrono>

#include <forge.h>
#define USE_FORGE_CUDA_COPY_HELPERS
#include <ComputeCopy.h>
#include <complex>
#include <cmath>
#include <vector>
#include <iostream>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <AdvectionDiffusionSolver1D.h>
#include <AdvectionDiffusionSolver2D.h>
#include <WaveEquationSolver1D.h>
#include <WaveEquationSolver2D.h>
#include <IterableEnum.h>

#pragma region Command Line Parser

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
	T GetArgumentValue(const std::string& option, const T& defaultValue) const noexcept
	{
		T ret;
		try
		{
			ret = GetArgumentValue<T>(option);
		}
		catch (int)
		{
			ret = defaultValue;
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
			std::abort();
		return *itr;
	}

	throw 42;
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

SpaceDiscretizerType parseSpaceDiscretizer(const std::string& text)
{
#define PARSE_WORKER(X) PARSE(SpaceDiscretizerType, X);

	PARSE_WORKER(Centered);
	PARSE_WORKER(Upwind);
	PARSE_WORKER(LaxWendroff);

#undef PARSE_WORKER
	return SpaceDiscretizerType::Null;
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

MathDomain parseMathDomain(const std::string& text)
{
#define PARSE_WORKER(X) PARSE(MathDomain, X);

	PARSE_WORKER(Double);
	PARSE_WORKER(Float);

#undef PARSE_WORKER
	return MathDomain::Null;
}

#undef PARSE

#pragma endregion

template<class solverImpl, MathDomain md>
void runner1D(const CommandLineArgumentParser& ap, const bool debug)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

#define DEBUG_PRINT_START(X)\
	if (debug)\
    {\
	    start = std::chrono::high_resolution_clock::now(); \
		std::cout << #X << std::endl;\
	}

#define DEBUG_PRINT_END\
	if (debug)\
	{\
		end = std::chrono::high_resolution_clock::now(); \
		double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();\
		std::cout << "... Done" << "[ " << elapsedTime << " ms ]" << std::endl;\
	}

#define DEBUG_PRINT(X)\
	if (debug)\
        std::cout << #X << std::endl;

	DEBUG_PRINT_START(Parsing inputs ...)

#pragma region Parse Inputs

	auto initialConditionFileString = ap.GetArgumentValue<std::string>("-ic", "");
	auto gridFileString = ap.GetArgumentValue<std::string>("-g", "");
	auto outputFileString = ap.GetArgumentValue<std::string>("-of", "sol.cl");

	auto leftBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-lbct");
	auto leftBoundaryConditionType = parseBoundaryConditionType(leftBoundaryConditionTypeString);
	double leftBoundaryConditionValue = ap.GetArgumentValue<double>("-lbc");

	std::string rightBoundaryConditionTypeString = ap.GetArgumentValue("-rbct", leftBoundaryConditionTypeString);
	auto rightBoundaryConditionType = parseBoundaryConditionType(rightBoundaryConditionTypeString);
	double rightBoundaryConditionValue = ap.GetArgumentValue("-rbc", leftBoundaryConditionValue);

	auto solverType = parseSolverType(ap.GetArgumentValue<std::string>("-st", "CrankNicolson"));
	auto spaceDiscretizerType = parseSpaceDiscretizer(ap.GetArgumentValue<std::string>("-sdt", "Upwind"));
	if (spaceDiscretizerType == SpaceDiscretizerType::LaxWendroff && solverType != SolverType::ExplicitEuler)
	{
		std::cout << "WARNING: Lax-Wendroff scheme can be applied only with ExplicitEuler -> overriding solver type" << std::endl;
		solverType = SolverType::ExplicitEuler;
	}

	auto diffusion = ap.GetArgumentValue<double>("-d");
	auto velocity = ap.GetArgumentValue<double>("-v");
	auto dt = ap.GetArgumentValue<double>("-dt");

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<int>("-n");

	// total number of steps
	size_t N = static_cast<size_t>(ap.GetArgumentValue<int>("-N"));

#pragma endregion

	DEBUG_PRINT_END;

	using vType = cl::Vector<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	DEBUG_PRINT_START(Creating grid...);

	std::ifstream gridFile(gridFileString);
	vType *grid = nullptr;
	if (!gridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128));
		grid = new vType(cl::LinSpace<MemorySpace::Device, md>(sType(-4.0), sType(4.0), 128u));
	}
	else
	{
		DEBUG_PRINT(... reading from file);
		grid = new vType(cl::VectorFromBinaryFile<MemorySpace::Device, md>(gridFileString));
	}
	DEBUG_PRINT_END;

	DEBUG_PRINT_START(Creating initial condition ...);

	std::ifstream initialConditionFile(initialConditionFileString);
	vType *initialCondition = nullptr;
	if (!initialConditionFile.is_open())
	{
		DEBUG_PRINT(... creating bell function);
		auto _grid = grid->Get();
		std::vector<sType> bellFunction(grid->size());
		for (unsigned i = 0; i < bellFunction.size(); ++i)
			bellFunction[i] = exp(-.25 * _grid[i] * _grid[i]);

		initialCondition = new vType(bellFunction);
	}
	else
	{
		DEBUG_PRINT(... reading from file);
		initialCondition = new vType(cl::VectorFromBinaryFile<MemorySpace::Device, md>(initialConditionFileString));
	}
	DEBUG_PRINT_END;

	BoundaryCondition leftBc(leftBoundaryConditionType, leftBoundaryConditionValue);
	BoundaryCondition rightBc(rightBoundaryConditionType, rightBoundaryConditionValue);
	BoundaryCondition1D bc(leftBc, rightBc);

	DEBUG_PRINT_START(Creating PDE input data ...);
	pde::PdeInputData1D<MemorySpace::Device, md> data(*initialCondition, *grid, velocity, diffusion, dt, solverType, spaceDiscretizerType, bc);
	DEBUG_PRINT_END;

	DEBUG_PRINT_START(Creating PDE solver ...);
	solverImpl solver(data);
	DEBUG_PRINT_END;

	std::vector<sType> solutionMatrix;

	DEBUG_PRINT_START(Solving ...);

	forge::Window wnd(1000, 800, "Plotting Demo");
	wnd.makeCurrent();

	forge::Chart chart(FG_CHART_2D);
	auto _grid = grid->Get();
	auto _initialCondition = initialCondition->Get();
	chart.setAxesLimits(_grid.front(), _grid.back(), *std::min_element(_initialCondition.begin(), _initialCondition.end()), *std::max_element(_initialCondition.begin(), _initialCondition.end()));

	static constexpr forge::dtype precision = forge::f32;
	forge::Plot plt = chart.plot(grid->size(), precision, FG_PLOT_LINE, FG_MARKER_NONE);
	plt.setColor(FG_BLUE);

	GfxHandle* handles;
	createGLBuffer(&handles, plt.vertices(), FORGE_VERTEX_BUFFER);

	bool toDo = true;
	cl::Vector<MemorySpace::Device, MathDomain::Float>* xyPair = nullptr;
	do
	{		
		if (toDo)
		{
			for (unsigned m = 0; m < N; ++m)
			{
				solver.Advance(n);

				if (!xyPair)
					xyPair = new cl::Vector<MemorySpace::Device, MathDomain::Float>(2 * grid->size());
				cl::MakePair(*xyPair, *grid, *solver.solution->columns[0]);
				copyToGLBuffer(handles, (ComputeResourceHandle)xyPair->GetBuffer().pointer, plt.verticesSize());
				wnd.draw(chart);
			}
		}

		wnd.draw(chart);
		toDo = false;
	}
	while (!wnd.close());
	releaseGLBuffer(handles);

	DEBUG_PRINT_END;

	delete initialCondition;
	delete grid;

#undef DEBUB_PRINT
}

template<class solverImpl, MathDomain md>
void runner2D(const CommandLineArgumentParser& ap, const bool debug)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

	DEBUG_PRINT_START(Parsing inputs ...)

#pragma region Parse Inputs

	auto initialConditionFileString = ap.GetArgumentValue<std::string>("-ic", "");
	auto xGridFileString = ap.GetArgumentValue<std::string>("-gx", "");
	auto yGridFileString = ap.GetArgumentValue<std::string>("-gy", "");
	auto outputFileString = ap.GetArgumentValue<std::string>("-of", "sol.cl");

#pragma region BC

	auto leftBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-lbct");
	auto leftBoundaryConditionType = parseBoundaryConditionType(leftBoundaryConditionTypeString);
	double leftBoundaryConditionValue = ap.GetArgumentValue<double>("-lbc");

	std::string rightBoundaryConditionTypeString = ap.GetArgumentValue("-rbct", leftBoundaryConditionTypeString);
	auto rightBoundaryConditionType = parseBoundaryConditionType(rightBoundaryConditionTypeString);
	double rightBoundaryConditionValue = ap.GetArgumentValue("-rbc", leftBoundaryConditionValue);

	auto downBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-dbct", leftBoundaryConditionTypeString);
	auto downBoundaryConditionType = parseBoundaryConditionType(downBoundaryConditionTypeString);
	auto downBoundaryConditionValue = ap.GetArgumentValue<double>("-dbc", leftBoundaryConditionValue);

	auto upBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-ubct", leftBoundaryConditionTypeString);
	auto upBoundaryConditionType = parseBoundaryConditionType(upBoundaryConditionTypeString);
    auto upBoundaryConditionValue = ap.GetArgumentValue<double>("-ubc", leftBoundaryConditionValue);

#pragma endregion

	auto solverType = parseSolverType(ap.GetArgumentValue<std::string>("-st", "CrankNicolson"));
	auto spaceDiscretizerType = parseSpaceDiscretizer(ap.GetArgumentValue<std::string>("-sdt", "Upwind"));
	if (spaceDiscretizerType == SpaceDiscretizerType::LaxWendroff && solverType != SolverType::ExplicitEuler)
	{
		std::cout << "WARNING: Lax-Wendroff scheme can be applied only with ExplicitEuler -> overriding solver type" << std::endl;
		solverType = SolverType::ExplicitEuler;
	}

	auto diffusion = ap.GetArgumentValue<double>("-d");
	auto xVelocity = ap.GetArgumentValue<double>("-vx");
	auto yVelocity = ap.GetArgumentValue<double>("-vy");
	auto dt = ap.GetArgumentValue<double>("-dt");

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<int>("-n");

	// total number of steps
	unsigned N = static_cast<unsigned>(ap.GetArgumentValue<int>("-N"));

#pragma endregion

	DEBUG_PRINT_END;

	using vType = cl::Vector<MemorySpace::Device, md>;
	using mType = cl::ColumnWiseMatrix<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

#pragma region Grid

	DEBUG_PRINT_START(Creating x grid...);

	std::ifstream xGridFile(xGridFileString);
	vType *xGrid = nullptr;
	constexpr unsigned defaultSize = 128u;
	if (!xGridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128));
		xGrid = new vType(cl::LinSpace<MemorySpace::Device, md>(sType(-4.0), sType(4.0), defaultSize));
	}
	else
	{
		DEBUG_PRINT(... reading from file);
		xGrid = new vType(cl::VectorFromBinaryFile<MemorySpace::Device, md>(xGridFileString));
	}
	DEBUG_PRINT_END;

	DEBUG_PRINT_START(Creating y grid...);

	std::ifstream yGridFile(yGridFileString);
	vType *yGrid = nullptr;
	if (!yGridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128));
		yGrid = new vType(cl::LinSpace<MemorySpace::Device, md>(sType(-4.0), sType(4.0), defaultSize));
	}
	else
	{
		DEBUG_PRINT(... reading from file);
		yGrid = new vType(cl::VectorFromBinaryFile<MemorySpace::Device, md>(yGridFileString));
	}
	DEBUG_PRINT_END;

#pragma endregion

	DEBUG_PRINT_START(Creating initial condition ...);

	std::ifstream initialConditionFile(initialConditionFileString);
	mType *initialCondition = nullptr;
	if (!initialConditionFile.is_open())
	{
		DEBUG_PRINT(... creating bell function);
		auto _xGrid = xGrid->Get();
		auto _yGrid = yGrid->Get();
		std::vector<sType> bellFunction(xGrid->size() * yGrid->size());
		for (unsigned j = 0; j < _yGrid.size(); ++j)
			for (unsigned i = 0; i < _xGrid.size(); ++i)
				bellFunction[i + _xGrid.size() * j] = exp(-.25 * (_xGrid[i] * _xGrid[i] + _yGrid[j] * _yGrid[j]));

		initialCondition = new mType(bellFunction, xGrid->size(), yGrid->size());
	}
	else
	{
		DEBUG_PRINT(... reading from file);
		initialCondition = new mType(cl::MatrixFromBinaryFile<MemorySpace::Device, md>(initialConditionFileString));
	}
	DEBUG_PRINT_END;

	BoundaryCondition leftBc(leftBoundaryConditionType, leftBoundaryConditionValue);
	BoundaryCondition rightBc(rightBoundaryConditionType, rightBoundaryConditionValue);
	BoundaryCondition downBc(downBoundaryConditionType, downBoundaryConditionValue);
	BoundaryCondition upBc(upBoundaryConditionType, upBoundaryConditionValue);
	BoundaryCondition2D bc(leftBc, rightBc, downBc, upBc);

	DEBUG_PRINT_START(Creating PDE input data ...);
	pde::PdeInputData2D<MemorySpace::Device, md> data(*initialCondition, *xGrid, *yGrid,
													  xVelocity, yVelocity, diffusion, dt, solverType, spaceDiscretizerType, bc);
	DEBUG_PRINT_END;

	DEBUG_PRINT_START(Creating PDE solver ...);
	solverImpl solver(data);
	DEBUG_PRINT_END;

	// solution matrix is a collection of flattened solutions over time
	forge::Window wnd(1024, 768, "3d Surface Demo");
	wnd.makeCurrent();

	forge::Chart chart(FG_CHART_3D);

	auto _xGrid = xGrid->Get();
	auto _yGrid = yGrid->Get();
	auto _ic = initialCondition->Get();
	chart.setAxesLimits(_xGrid.front(), _xGrid.back(), _yGrid.front(), _yGrid.back(), *std::min_element(_ic.begin(), _ic.end()), *std::max_element(_ic.begin(), _ic.end()));
	chart.setAxesTitles("x-axis", "y-axis", "z-axis");

	forge::Surface surf = chart.surface(_xGrid.size(), _yGrid.size(), forge::f32);
	surf.setColor(FG_BLUE);

	GfxHandle* handle;
	createGLBuffer(&handle, surf.vertices(), FORGE_VERTEX_BUFFER);

	bool toDo = true;
	cl::Vector<MemorySpace::Device, MathDomain::Float>* xyzTriple = nullptr;
	do
	{
		if (toDo)
		{
			for (unsigned m = 0; m < N; ++m)
			{
				solver.Advance(n);

				if (!xyzTriple)
					xyzTriple = new cl::Vector<MemorySpace::Device, MathDomain::Float>(3 * xGrid->size() * yGrid->size());

				cl::MakeTriple(*xyzTriple, *xGrid, *yGrid, *solver.solution->columns[0]);
				copyToGLBuffer(handle, (ComputeResourceHandle)xyzTriple->GetBuffer().pointer, surf.verticesSize());
				wnd.draw(chart);
			}
		}

		wnd.draw(chart);
		toDo = false;
	}
	while (!wnd.close());
	releaseGLBuffer(handle);

	delete initialCondition;
	delete xGrid;
	delete yGrid;

#undef DEBUB_PRINT
}


int main(int argc, char** argv)
{
	CommandLineArgumentParser ap(argc, argv);

	auto mathDomain = parseMathDomain(ap.GetArgumentValue<std::string>("-md", "Float"));
	auto pdeType = ap.GetArgumentValue<std::string>("-pde", "AdvectionDiffusion");
	auto dimensionality = ap.GetArgumentValue<int>("-dim", 1);
	auto debug = ap.GetFlag("-dbg");

	if (dimensionality == 1)
	{
		switch (mathDomain)
		{
			case MathDomain::Float:
				if (pdeType == "AdvectionDiffusion")
					runner1D<pde::AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
				else if (pdeType == "WaveEquation")
					runner1D<pde::WaveEquationSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
				if (pdeType == "AdvectionDiffusion")
					runner1D<pde::AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);
				else if (pdeType == "WaveEquation")
					runner1D<pde::WaveEquationSolver1D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);
				else
					throw NotImplementedException();
				break;
			default:
				throw NotImplementedException();
		}
	}
	else if (dimensionality == 2)
	{
		switch (mathDomain)
		{
			case MathDomain::Float:
				if (pdeType == "AdvectionDiffusion")
					runner2D<pde::AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
				else if (pdeType == "WaveEquation")
					runner2D<pde::WaveEquationSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
				if (pdeType == "AdvectionDiffusion")
					runner2D<pde::AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);
				else if (pdeType == "WaveEquation")
					runner2D<pde::WaveEquationSolver2D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);  // FIXME
				else
					throw NotImplementedException();
				break;
			default:
				throw NotImplementedException();
		}
	}

	return 0;
}

